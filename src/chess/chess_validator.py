from __future__ import annotations
import copy
import re
from dataclasses import dataclass, field
from typing import Optional

"""
chess_validator.py
------------------
A self-contained chess engine for validating SAN moves and inspecting board state.
Designed to be plugged into a model pipeline for move verification.

Usage:
    game = ChessGame()
    ok, msg = game.push_san("e4")      # returns (True, "e4") or (False, "error msg")
    ok, msg = game.push_san("e5")
    print(game.board_string())          # ASCII board
    print(game.state_dict())            # full machine-readable state
    print(game.pgn())                   # PGN string of the game so far
"""

"""
2. Validate a model's proposed move before applying it:

pythonreport = validate_single_move("Nf3", game)
# report["valid"], report["canonical_san"], report["legal_moves"]

3. Validate an entire sequence at once:
pythonreport = validate_move_sequence(["e4", "e5", "Nf3", "Nc6", "Bc4"])
# report["all_valid"], report["move_results"], report["pgn"], report["fen"]

What state_dict() gives the model:
Key                         Contents
fen                         Full FEN string
turn / turn_name            "w" / "White"
in_check                    True/False
legal_moves                 Every legal SAN move for the side to move
pieces                      Dict of square → {color, type, piece}
move_history                All moves so far as canonical SAN
castling_rights             Rights for all four castles
en_passant_target           Square name or null
result                      "1-0", "0-1", "1/2-1/2", or null

Supports all rules: castling, en passant, promotion, check/checkmate detection, stalemate, 
and modified this to 100 -> (the 50-move draw rule).
"""

# ── Constants ────────────────────────────────────────────────────────────────

FILES = "abcdefgh"
PIECE_TYPES = {"K", "Q", "R", "B", "N", "P"}
COLORS = {"w", "b"}

UNICODE = {
    "wK": "♔", "wQ": "♕", "wR": "♖", "wB": "♗", "wN": "♘", "wP": "♙",
    "bK": "♚", "bQ": "♛", "bR": "♜", "bB": "♝", "bN": "♞", "bP": "♟",
}

INIT_BOARD: list[list[Optional[str]]] = [
    ["bR","bN","bB","bQ","bK","bB","bN","bR"],
    ["bP","bP","bP","bP","bP","bP","bP","bP"],
    [None]*8, [None]*8, [None]*8, [None]*8,
    ["wP","wP","wP","wP","wP","wP","wP","wP"],
    ["wR","wN","wB","wQ","wK","wB","wN","wR"],
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def pc(p: str) -> str:
    """Return color of piece string, e.g. 'wN' -> 'w'."""
    return p[0]

def pt(p: str) -> str:
    """Return type of piece string, e.g. 'wN' -> 'N'."""
    return p[1]

def opp(c: str) -> str:
    return "b" if c == "w" else "w"

def sq_name(r: int, f: int) -> str:
    return FILES[f] + str(8 - r)

def sq_from_name(name: str) -> tuple[int, int]:
    return 8 - int(name[1]), FILES.index(name[0])

# ── Move dataclass ────────────────────────────────────────────────────────────

@dataclass
class Move:
    fr: int
    ff: int
    tr: int
    tf: int
    promo: Optional[str] = None      # 'Q','R','B','N'
    castle: Optional[str] = None     # 'K' or 'Q'
    ep: bool = False                 # en passant capture
    double_push: bool = False        # pawn double-push
    san: str = ""                    # filled after legality check


# ── Core board logic ──────────────────────────────────────────────────────────

def pseudo_moves(
    board: list[list[Optional[str]]],
    r: int, f: int,
    ep: Optional[tuple[int,int]],
    castling: dict[str,bool],
) -> list[Move]:
    """Generate pseudo-legal moves (may leave king in check)."""
    p = board[r][f]
    if not p:
        return []
    c, t = pc(p), pt(p)
    moves: list[Move] = []

    def add(nr, nf, **kw):
        if 0 <= nr < 8 and 0 <= nf < 8:
            moves.append(Move(r, f, nr, nf, **kw))

    if t == "P":
        d = -1 if c == "w" else 1
        start_r = 6 if c == "w" else 1
        # Forward
        if not board[r+d][f]:
            add(r+d, f)
            if r == start_r and not board[r+2*d][f]:
                add(r+2*d, f, double_push=True)
        # Captures
        for df in (-1, 1):
            nr, nf = r+d, f+df
            if 0 <= nr < 8 and 0 <= nf < 8:
                target = board[nr][nf]
                if target and pc(target) == opp(c):
                    add(nr, nf)
                elif ep and (nr, nf) == ep:
                    add(nr, nf, ep=True)

    elif t == "N":
        for dr, df in ((2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)):
            nr, nf = r+dr, f+df
            if 0 <= nr < 8 and 0 <= nf < 8:
                tgt = board[nr][nf]
                if not tgt or pc(tgt) == opp(c):
                    add(nr, nf)

    else:
        rays = []
        if t in ("B", "Q"):
            rays += [(1,1),(1,-1),(-1,1),(-1,-1)]
        if t in ("R", "Q"):
            rays += [(1,0),(-1,0),(0,1),(0,-1)]
        for dr, df in rays:
            nr, nf = r+dr, f+df
            while 0 <= nr < 8 and 0 <= nf < 8:
                tgt = board[nr][nf]
                if tgt:
                    if pc(tgt) == opp(c):
                        add(nr, nf)
                    break
                add(nr, nf)
                nr += dr; nf += df

        if t == "K":
            for dr, df in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                nr, nf = r+dr, f+df
                if 0 <= nr < 8 and 0 <= nf < 8:
                    tgt = board[nr][nf]
                    if not tgt or pc(tgt) == opp(c):
                        add(nr, nf)
            # Castling
            no_cast = {k:False for k in castling}
            if c == "w" and r == 7 and f == 4:
                if castling.get("wK") and not board[7][5] and not board[7][6]:
                    if not sq_attacked(board,7,4,"b",no_cast) and not sq_attacked(board,7,5,"b",no_cast):
                        add(7, 6, castle="K")
                if castling.get("wQ") and not board[7][3] and not board[7][2] and not board[7][1]:
                    if not sq_attacked(board,7,4,"b",no_cast) and not sq_attacked(board,7,3,"b",no_cast):
                        add(7, 2, castle="Q")
            if c == "b" and r == 0 and f == 4:
                if castling.get("bK") and not board[0][5] and not board[0][6]:
                    if not sq_attacked(board,0,4,"w",no_cast) and not sq_attacked(board,0,5,"w",no_cast):
                        add(0, 6, castle="K")
                if castling.get("bQ") and not board[0][3] and not board[0][2] and not board[0][1]:
                    if not sq_attacked(board,0,4,"w",no_cast) and not sq_attacked(board,0,3,"w",no_cast):
                        add(0, 2, castle="Q")
    return moves


def sq_attacked(
    board: list[list[Optional[str]]],
    r: int, f: int, by: str,
    castling: dict[str,bool],
) -> bool:
    """Return True if square (r,f) is attacked by color `by`."""
    no_ep: Optional[tuple] = None
    for pr in range(8):
        for pf in range(8):
            p = board[pr][pf]
            if p and pc(p) == by:
                for m in pseudo_moves(board, pr, pf, no_ep, castling):
                    if m.tr == r and m.tf == f:
                        return True
    return False


def find_king(board: list[list[Optional[str]]], c: str) -> Optional[tuple[int,int]]:
    for r in range(8):
        for f in range(8):
            p = board[r][f]
            if p and pc(p) == c and pt(p) == "K":
                return (r, f)
    return None


def in_check(board: list[list[Optional[str]]], c: str) -> bool:
    k = find_king(board, c)
    if not k:
        return False
    return sq_attacked(board, k[0], k[1], opp(c), {})


def apply_move(
    board: list[list[Optional[str]]],
    m: Move,
    castling: dict[str,bool],
    promo: Optional[str] = None,
) -> list[list[Optional[str]]]:
    """Return a new board with the move applied (does not mutate input)."""
    nb = copy.deepcopy(board)
    p = nb[m.fr][m.ff]
    if m.castle:
        r = m.fr
        nb[r][m.tf] = p; nb[m.fr][m.ff] = None
        if m.castle == "K":
            nb[r][5] = nb[r][7]; nb[r][7] = None
        else:
            nb[r][3] = nb[r][0]; nb[r][0] = None
    else:
        if m.ep:
            ep_r = m.fr
            nb[ep_r][m.tf] = None
        nb[m.tr][m.tf] = p; nb[m.fr][m.ff] = None
        if pt(p) == "P" and m.tr in (0, 7):
            nb[m.tr][m.tf] = pc(p) + (promo or "Q")
    return nb


def legal_moves(
    board: list[list[Optional[str]]],
    r: int, f: int,
    ep: Optional[tuple[int,int]],
    castling: dict[str,bool],
    turn: str,
) -> list[Move]:
    """Return fully legal moves for the piece at (r,f)."""
    p = board[r][f]
    if not p or pc(p) != turn:
        return []
    c = pc(p)
    result = []
    for m in pseudo_moves(board, r, f, ep, castling):
        nb = apply_move(board, m, castling)
        if not in_check(nb, c):
            result.append(m)
    return result


def all_legal_moves(
    board: list[list[Optional[str]]],
    turn: str,
    ep: Optional[tuple[int,int]],
    castling: dict[str,bool],
) -> list[Move]:
    """All legal moves for the side to move."""
    moves = []
    for r in range(8):
        for f in range(8):
            p = board[r][f]
            if p and pc(p) == turn:
                moves.extend(legal_moves(board, r, f, ep, castling, turn))
    return moves


# ── SAN parsing ───────────────────────────────────────────────────────────────

def parse_san(
    san: str,
    board: list[list[Optional[str]]],
    turn: str,
    ep: Optional[tuple[int,int]],
    castling: dict[str,bool],
) -> Optional[tuple[Move, Optional[str]]]:
    """
    Parse a SAN string and return (Move, promo_piece) or None if illegal.
    promo_piece is 'Q','R','B','N' or None.
    """
    raw = san.strip()
    # Strip check/mate/annotation suffixes
    clean = re.sub(r"[+#!?]*$", "", raw)

    # Castling
    if clean in ("O-O", "0-0"):
        for m in all_legal_moves(board, turn, ep, castling):
            if m.castle == "K":
                return (m, None)
        return None
    if clean in ("O-O-O", "0-0-0"):
        for m in all_legal_moves(board, turn, ep, castling):
            if m.castle == "Q":
                return (m, None)
        return None

    # Promotion
    promo = None
    promo_match = re.search(r"=?([QRBN])$", clean)
    if promo_match:
        promo = promo_match.group(1)
        clean = clean[: promo_match.start()]

    # Determine piece type
    piece_type = "P"
    if clean and clean[0] in "KQRBN":
        piece_type = clean[0]
        clean = clean[1:]

    # Strip capture 'x'
    clean = clean.replace("x", "")

    # Parse destination + optional disambiguation
    dest_file = dest_rank = src_file = src_rank = None
    try:
        if len(clean) == 2:
            dest_file = FILES.index(clean[0])
            dest_rank = 8 - int(clean[1])
        elif len(clean) == 3:
            if clean[0] in FILES:
                src_file = FILES.index(clean[0])
            elif clean[0].isdigit():
                src_rank = 8 - int(clean[0])
            dest_file = FILES.index(clean[1])
            dest_rank = 8 - int(clean[2])
        elif len(clean) == 4:
            src_file = FILES.index(clean[0])
            src_rank = 8 - int(clean[1])
            dest_file = FILES.index(clean[2])
            dest_rank = 8 - int(clean[3])
        else:
            return None
    except (ValueError, IndexError):
        return None

    if dest_file is None or dest_rank is None:
        return None
    if not (0 <= dest_file <= 7 and 0 <= dest_rank <= 7):
        return None

    candidates = []
    for m in all_legal_moves(board, turn, ep, castling):
        p = board[m.fr][m.ff]
        if not p or pt(p) != piece_type:
            continue
        if m.tr != dest_rank or m.tf != dest_file:
            continue
        if src_file is not None and m.ff != src_file:
            continue
        if src_rank is not None and m.fr != src_rank:
            continue
        # Promotion: if promo specified, only allow matching; if not specified, filter out promo moves
        is_promo_move = (piece_type == "P" and dest_rank in (0, 7))
        if is_promo_move and not promo:
            promo = "Q"  # default to queen if not specified
        candidates.append(m)

    if len(candidates) == 1:
        return (candidates[0], promo)
    elif len(candidates) > 1:
        return None  # ambiguous
    return None


# ── SAN generation ────────────────────────────────────────────────────────────

def move_to_san(
    board: list[list[Optional[str]]],
    m: Move,
    promo: Optional[str],
    ep: Optional[tuple[int,int]],
    castling: dict[str,bool],
) -> str:
    """Generate SAN string for a move (before applying it)."""
    if m.castle:
        return "O-O" if m.castle == "K" else "O-O-O"

    p = board[m.fr][m.ff]
    c, t = pc(p), pt(p)
    san = "" if t == "P" else t

    # Disambiguation
    ambig = []
    for am in all_legal_moves(board, c, ep, castling):
        ap = board[am.fr][am.ff]
        if ap and pt(ap) == t and (am.fr != m.fr or am.ff != m.ff):
            if am.tr == m.tr and am.tf == m.tf:
                ambig.append(am)

    if ambig:
        same_file = [a for a in ambig if a.ff == m.ff]
        same_rank = [a for a in ambig if a.fr == m.fr]
        if not same_file:
            san += FILES[m.ff]
        elif not same_rank:
            san += str(8 - m.fr)
        else:
            san += FILES[m.ff] + str(8 - m.fr)
    elif t == "P" and (board[m.tr][m.tf] or m.ep):
        san += FILES[m.ff]  # pawn captures always include origin file

    # Capture
    if board[m.tr][m.tf] or m.ep:
        san += "x"

    san += sq_name(m.tr, m.tf)

    # Promotion
    if promo:
        san += "=" + promo

    # Check / checkmate
    nb = apply_move(board, m, castling, promo)
    next_c = opp(c)
    if in_check(nb, next_c):
        has_any = bool(all_legal_moves(nb, next_c, None, {k: False for k in castling}))
        san += "+" if has_any else "#"

    return san


# ── Main game class ───────────────────────────────────────────────────────────

@dataclass
class ChessGame:
    """
    Stateful chess game. Use push_san() to make moves.
    All public methods are safe to call at any time.
    """
    board: list[list[Optional[str]]] = field(default_factory=lambda: copy.deepcopy(INIT_BOARD))
    turn: str = "w"
    castling: dict[str, bool] = field(default_factory=lambda: {
        "wK": True, "wQ": True, "bK": True, "bQ": True
    })
    en_passant: Optional[tuple[int, int]] = None
    move_list: list[str] = field(default_factory=list)   # SAN strings in order
    halfmove_clock: int = 0   # for 50-move rule
    fullmove_number: int = 1
    result: Optional[str] = None  # '1-0', '0-1', '1/2-1/2', or None

    # ── Push a move ──────────────────────────────────────────────────────────

    def push_san(self, san: str) -> tuple[bool, str]:
        """
        Attempt to apply a SAN move.
        Returns (True, canonical_san) on success.
        Returns (False, error_message) on failure.
        """
        if self.result:
            return False, f"Game is already over: {self.result}"

        parsed = parse_san(san, self.board, self.turn, self.en_passant, self.castling)
        if parsed is None:
            return False, f'Illegal or ambiguous move: "{san}" for {self.turn} to move'

        m, promo = parsed
        canonical = move_to_san(self.board, m, promo, self.en_passant, self.castling)

        # Update castling rights
        p = self.board[m.fr][m.ff]
        if p:
            if pt(p) == "K":
                self.castling[self.turn + "K"] = False
                self.castling[self.turn + "Q"] = False
            if pt(p) == "R":
                if m.ff == 0: self.castling[self.turn + "Q"] = False
                if m.ff == 7: self.castling[self.turn + "K"] = False
        if self.board[m.tr][m.tf]:
            tgt = self.board[m.tr][m.tf]
            if pt(tgt) == "R":
                opc = opp(self.turn)
                if m.tf == 0: self.castling[opc + "Q"] = False
                if m.tf == 7: self.castling[opc + "K"] = False

        # En passant target for next move
        self.en_passant = (m.fr + (-1 if self.turn == "w" else 1), m.ff) if m.double_push else None

        # Halfmove clock
        captured = self.board[m.tr][m.tf] is not None or m.ep
        pawn_move = pt(p) == "P" if p else False
        self.halfmove_clock = 0 if (captured or pawn_move) else self.halfmove_clock + 1

        # Apply move
        self.board = apply_move(self.board, m, self.castling, promo)
        self.move_list.append(canonical)

        if self.turn == "b":
            self.fullmove_number += 1
        self.turn = opp(self.turn)

        # Check game termination
        self._check_termination()

        return True, canonical

    # ── Verify move without applying ─────────────────────────────────────────

    def verify_san(self, san: str) -> tuple[bool, str]:
        """
        Check if a SAN move is legal WITHOUT applying it.
        Returns (True, canonical_san) or (False, reason).
        """
        if self.result:
            return False, f"Game is already over: {self.result}"
        parsed = parse_san(san, self.board, self.turn, self.en_passant, self.castling)
        if parsed is None:
            return False, f'Illegal or ambiguous move: "{san}"'
        m, promo = parsed
        canonical = move_to_san(self.board, m, promo, self.en_passant, self.castling)
        return True, canonical

    # ── Board display ─────────────────────────────────────────────────────────

    def board_string(self, unicode: bool = True) -> str:
        """Return a printable ASCII/unicode board."""
        lines = []
        lines.append("  +" + "---+" * 8)
        for r in range(8):
            row = f"{8-r} |"
            for f in range(8):
                p = self.board[r][f]
                if p:
                    sym = UNICODE[p] if unicode else p
                    row += f" {sym} |"
                else:
                    row += "   |"
            lines.append(row)
            lines.append("  +" + "---+" * 8)
        lines.append("    " + "   ".join(FILES))
        return "\n".join(lines)

    # ── State dict (for model consumption) ───────────────────────────────────

    def state_dict(self) -> dict:
        """
        Return a fully machine-readable snapshot of the game state.
        Designed for feeding directly into a model.
        """
        pieces = {}
        for r in range(8):
            for f in range(8):
                p = self.board[r][f]
                if p:
                    sq = sq_name(r, f)
                    pieces[sq] = {"color": pc(p), "type": pt(p), "piece": p}

        legal = []
        for m in all_legal_moves(self.board, self.turn, self.en_passant, self.castling):
            promo_opts = ["Q","R","B","N"] if (
                pt(self.board[m.fr][m.ff]) == "P" and m.tr in (0,7)
            ) else [None]
            for promo in promo_opts:
                s = move_to_san(self.board, m, promo, self.en_passant, self.castling)
                legal.append(s)

        return {
            "fen": self.fen(),
            "turn": self.turn,
            "turn_name": "White" if self.turn == "w" else "Black",
            "move_number": self.fullmove_number,
            "halfmove_clock": self.halfmove_clock,
            "in_check": in_check(self.board, self.turn),
            "castling": dict(self.castling),
            "en_passant_target": sq_name(*self.en_passant) if self.en_passant else None,
            "pieces": pieces,
            "legal_moves": sorted(legal),
            "move_history": list(self.move_list),
            "pgn_moves": self.pgn(),
            "result": self.result,
            "game_over": self.result is not None,
        }

    # ── FEN ───────────────────────────────────────────────────────────────────

    def fen(self) -> str:
        """Return the FEN string for the current position."""
        fen_rows = []
        for r in range(8):
            empty = 0
            row_str = ""
            for f in range(8):
                p = self.board[r][f]
                if p:
                    if empty:
                        row_str += str(empty)
                        empty = 0
                    c, t = pc(p), pt(p)
                    row_str += t if c == "w" else t.lower()
                else:
                    empty += 1
            if empty:
                row_str += str(empty)
            fen_rows.append(row_str)

        board_fen = "/".join(fen_rows)
        turn_fen = self.turn
        cast = ""
        for k, v in [("wK","K"),("wQ","Q"),("bK","k"),("bQ","q")]:
            if self.castling.get(k): cast += v
        cast = cast or "-"
        ep = sq_name(*self.en_passant) if self.en_passant else "-"
        return f"{board_fen} {turn_fen} {cast} {ep} {self.halfmove_clock} {self.fullmove_number}"

    # ── PGN ───────────────────────────────────────────────────────────────────

    def pgn(self) -> str:
        """Return the move list in PGN format."""
        tokens = []
        for i, san in enumerate(self.move_list):
            if i % 2 == 0:
                tokens.append(f"{i//2 + 1}.")
            tokens.append(san)
        if self.result:
            tokens.append(self.result)
        return " ".join(tokens)

    # ── Termination check ─────────────────────────────────────────────────────

    def _check_termination(self):
        has_moves = bool(all_legal_moves(self.board, self.turn, self.en_passant, self.castling))
        if not has_moves:
            if in_check(self.board, self.turn):
                self.result = "1-0" if self.turn == "b" else "0-1"
            else:
                self.result = "1/2-1/2"
        elif self.halfmove_clock >= 200:
            self.result = "1/2-1/2"

    # ── Convenience ──────────────────────────────────────────────────────────

    def reset(self):
        """Reset to the starting position."""
        self.__init__()

    def push_moves(self, moves: list[str]) -> list[tuple[bool, str]]:
        """Push a list of SAN moves. Stops on first illegal move."""
        results = []
        for san in moves:
            ok, msg = self.push_san(san)
            results.append((ok, msg))
            if not ok:
                break
        return results

    def __repr__(self):
        return (
            f"ChessGame(turn={self.turn!r}, move={self.fullmove_number}, "
            f"moves={len(self.move_list)}, result={self.result!r})"
        )


# ── Model-oriented validator function ────────────────────────────────────────

def validate_move_sequence(moves: list[str]) -> dict:
    """
    Validate a full sequence of SAN moves from the starting position.
    Returns a report dict suitable for feeding to a model.

    Example:
        report = validate_move_sequence(["e4","e5","Nf3","Nc6","Bc4","Bc5","Qxf7"])
    """
    game = ChessGame()
    results = []
    for i, san in enumerate(moves):
        ok, msg = game.push_san(san)
        results.append({
            "move_index": i,
            "input_san": san,
            "canonical_san": msg if ok else None,
            "valid": ok,
            "error": None if ok else msg,
        })
        if not ok:
            break

    return {
        "all_valid": all(r["valid"] for r in results),
        "moves_validated": len(results),
        "move_results": results,
        "final_state": game.state_dict() if results and results[-1]["valid"] else None,
        "pgn": game.pgn(),
        "fen": game.fen(),
    }


def validate_single_move(san: str, game: ChessGame) -> dict:
    """
    Validate one SAN move against a live ChessGame instance without applying it.
    Useful for checking a model's proposed next move before committing.

    Example:
        game = ChessGame()
        game.push_san("e4")
        report = validate_single_move("e5", game)
    """
    ok, msg = game.verify_san(san)
    state = game.state_dict()
    return {
        "input_san": san,
        "valid": ok,
        "canonical_san": msg if ok else None,
        "error": None if ok else msg,
        "turn": state["turn"],
        "turn_name": state["turn_name"],
        "in_check": state["in_check"],
        "legal_moves": state["legal_moves"],
    }


# ── Demo / quick test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Demo 1 — Scholar's Mate")
    print("=" * 60)
    game = ChessGame()
    for san in ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6??", "Qxf7"]:
        ok, msg = game.push_san(san)
        status = f"✓ {msg}" if ok else f"✗ {msg}"
        print(f"  {san:12s} → {status}")
    print()
    print(game.board_string())
    print()
    print("PGN :", game.pgn())
    print("FEN :", game.fen())
    print("Result:", game.result)

    print()
    print("=" * 60)
    print("Demo 2 — validate_move_sequence")
    print("=" * 60)
    import json
    report = validate_move_sequence(["e4", "e5", "Nf3", "INVALID", "Bc4"])
    print(json.dumps({k: v for k, v in report.items() if k != "final_state"}, indent=2))

    print()
    print("=" * 60)
    print("Demo 3 — validate_single_move (checking model's proposed move)")
    print("=" * 60)
    game2 = ChessGame()
    game2.push_san("e4")
    for candidate in ["e5", "Qh4", "zz9"]:
        r = validate_single_move(candidate, game2)
        print(f"  '{candidate}': valid={r['valid']}, canonical={r['canonical_san']}, error={r['error']}")

    print()
    print("=" * 60)
    print("Demo 4 — state_dict keys")
    print("=" * 60)
    game3 = ChessGame()
    game3.push_san("e4")
    game3.push_san("e5")
    s = game3.state_dict()
    for k, v in s.items():
        if k not in ("pieces", "legal_moves"):
            print(f"  {k}: {v}")
    print(f"  pieces: {len(s['pieces'])} pieces on board")
    print(f"  legal_moves ({len(s['legal_moves'])}): {s['legal_moves'][:6]} ...")