import bz2
import enum
import json
import random
import pandas as pd
import re

import torch
from transformers import AutoTokenizer

class TrainingType(enum.Enum):
    # Next Move Prediction: The model is trained to predict the next move given a sequence of moves
    SUPERVISED_LEARNING = "sl"
    # Reinforcement Learning: The model is trained using reinforcement learning techniques using rewards
    REINFORCEMENT_LEARNING = "rl"

max_paraquet_files_per_training_type = 2
model_path = "C:\\Users\\DebashisDas\\personal\\models\\Qwen"
df = pd.DataFrame(columns=['input_ids', 'attention_mask'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path)

def prompt_generator(chess_json, training_type=TrainingType.SUPERVISED_LEARNING):
    # move extraction logic for supervised learning
    def parse_moves(moves, play_as, num_moves_to_parse):
        """Parse PGN into list of (move_no, white, black) tuples."""
        # Remove result at end
        moves_clean = re.sub(r'\s*(1-0|0-1|1/2-1/2|\*)\s*$', '', moves.strip())
        # Match: move_number. white_move [black_move]
        pattern = r'(\d+)\.\s+(\S+)(?:\s+(?!\d+\.)(\S+))?'
        extracted_moves = ""
        for m in re.finditer(pattern, moves_clean):
            move_no = int(m.group(1))
            white   = m.group(2)
            black   = m.group(3)  # None if Black didn't play (resignation)
            if play_as == "white" and move_no > num_moves_to_parse:
                break
            if play_as == "black" and move_no > num_moves_to_parse:
                extracted_moves += f"{move_no}. {white} "
                break
            extracted_moves += f"{move_no}. {white} {black if black else ''} "
        return extracted_moves
    # Generate the prompt based on the input JSON
    prompt = f"""
    Chess is a two-player abstract strategy board game played on an 8×8 grid (64 squares). Each player controls 16 pieces: one king, one queen, two rooks, two bishops, two knights, and eight pawns. White moves first, and the goal is to checkmate the opponent's king (threaten it with inescapable capture).
    Piece Movement

    King (K): One square in any direction; can castle once per game.
    Queen (Q): Any number of squares along a rank, file, or diagonal.
    Rook (R): Any number of squares along a rank or file.
    Bishop (B): Any number of squares diagonally.
    Knight (N): An "L"-shape — two squares in one direction, one in another; leaps over other pieces.
    Pawn: One square forward (or two on its first move); captures diagonally. Special moves: en passant and promotion.

    Standard Algebraic Notation (SAN) — PRIMARY FOCUS
    SAN is the universal standard for recording chess moves. Files are labeled a–h (left to right from White's perspective), and ranks are labeled 1–8 (bottom to top from White's perspective). Every square has a unique coordinate (e.g., e4, g7).
    Piece symbols: K = King, Q = Queen, R = Rook, B = Bishop, N = Knight. Pawns have no letter prefix.
    Basic move format:
    [Piece][destination square]

    Qg5 — Queen moves to g5
    e4 — Pawn moves to e4
    Nf3 — Knight moves to f3
    Bc4 — Bishop moves to c4

    Captures — insert x before the destination:
    [Piece]x[destination square]

    Bxf3 — Bishop captures on f3
    exd5 — Pawn on the e-file captures on d5 (pawn captures always include the origin file)
    Nxe5 — Knight captures on e5

    Disambiguation — when two identical pieces can reach the same square:
    [Piece][file or rank]x?[destination]

    Ngf3 — Knight from the g-file moves to f3
    R1e2 — Rook on rank 1 moves to e2
    Qh4xe1 — Queen on h4 captures on e1 (both file and rank given when necessary)

    Special moves:

    Kingside castling : 0-0
    Queenside castling : 0-0-0
    Pawn promotion : e8=Q or e8Q (piece chosen after =)
    Check : + suffix (e.g., Qd7+)
    Checkmate : # suffix (e.g., Qxf7#)

    Result formats:
    1-0 : White wins or black resigns mid game due to a losing position
    0-1 : Black wins or white resigns mid game due to a losing position
    1/2-1/2 : Draw

    Main Objective:

    The main objective of chess is to checkmate your opponent's king. This means putting the opponent's king in a position where it is under attack (in "check") and 
    there is no legal move to escape the attack. The game can also end in a draw if neither player can force a checkmate, if both players agree to a draw, 
    or if certain conditions are met (e.g., stalemate, threefold repetition, fifty-move rule). Sometimes players may resign if they believe they are in a losing position, 
    which also ends the game. 
    
    Understanding these patterns is the main objective of chess and the key to mastering the game.

    Example game in SAN format:
    1. e4 Nf6 2. e5 Nd5 3. d4 d6 4. c4 Nb6 5. exd6 cxd6 6. Nf3 g6 7. a4 Bg7 8. a5 Nb6d7 9. Bd2 Nc6 10. Bc3 O-O 11. Be2 b6 12. axb6 Nxb6 13. d5 Ne5 14. Nxe5 dxe5 15. c5 Nxd5 16. Bf3 Bb7 17. O-O f5 18. Bxd5+ Qxd5 19. Qxd5+ Bxd5 20. Na3 Rfc8 21. Nb5 Rxc5 22. Na3 e4 23. Nc2 Bxc3 24. bxc3 Rxc3 25. Ne3 e6 26. Nxd5 exd5 27. Rac1 Rxc1 28. Rxc1 d4 29. h3 d3 30. Kf1 a5 31. Ke1 a4 32. Ra1 a3 33. Kd2 Ra4 34. Ke3 Kf7 35. f3 Ke6 36. f4 Kd5 37. g3 Kc4 38. g4 Kc3 39. gxf5 gxf5 40. Rc1+ Kb2 41. Rc5 a2 42. Rb5+ Kc2 43. Rc5+ Kd1 44. Rxf5 a1=Q 45. Kf2 Qc1 46. Kg3 Qe3+ 47. Kh4 Qf2+ 48. Kg5 1-0
    
    In the above game both sides playes 48 moves and white won the game (1-0). The moves are in SAN format with move number and dot separator between white and black moves.

    You are a chess player playing as white or black based on "play_as" key and try to defeat your opponent. The moves will be in SAN format. 
    Your task it to predict the next set of moves for both black and white till you win the game. 
    The moves will be provided in the following format:
    {{
        "play_as": "<tells you whether to play as white or black>",
        "moves": "Moves in SAN format with move number and dot separator between white and black moves. For example : 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6",
    }}
    """
    move_numbers = re.findall(r'(\d+)\.', chess_json["moves"])
    num_moves = int(move_numbers[-1])
    if num_moves < 5:  # If the game has less than 10 moves, parse all moves
        raise ValueError(f"Game has only {num_moves} moves, which is less than the minimum required 10 moves for training.")
    num_moves_to_parse = random.randint((num_moves // 4), (num_moves // 4) * 3)  # Randomly choose to parse between 25% and 75% of the moves
    extracted_moves = parse_moves(chess_json["moves"], chess_json["play_as"], num_moves_to_parse=num_moves_to_parse)
    input_json = {
        "play_as": chess_json["play_as"],
        "moves": extracted_moves
    }
    if training_type == TrainingType.SUPERVISED_LEARNING:
        prompt += f"""
        Input JSON:
        {json.dumps(input_json, indent=4)}

        Output should be generated in the same input format after the moves provided as input

        ``` Output JSON ``` 
        {json.dumps(chess_json, indent=4)}
        ``` Output JSON END```
        """
    if training_type == TrainingType.REINFORCEMENT_LEARNING:
        prompt += f"""
        Input JSON:
        {json.dumps(input_json, indent=4)}

        Output should be generated in the same input format after the moves provided as input

        ``` Output JSON ``` 
        """
    return prompt

def open_bz2_file(file_path, counter, parquet_counter, paraquet_limit, training_type=TrainingType.SUPERVISED_LEARNING):
    global df, tokenizer
    counter = 0
    with bz2.open(file_path, 'rt', encoding='utf-8') as file:
        for i, line in enumerate(file):
            line = line.strip()
            if len(line) > 0 and not line.startswith("[") and "eval" not in line:  # Limit the number of lines printed
                chess_format_dic = {}
                chess_format_dic["moves"] = line
                lines = line.split(" ")
                if lines[-1].strip() == "1-0":
                    chess_format_dic["play_as"] = "white"
                elif lines[-1].strip() == "0-1":
                    chess_format_dic["play_as"] = "black"
                elif lines[-1].strip() == "1/2-1/2":
                    chess_format_dic["play_as"] = random.choice(["white", "black"])
                else:
                    continue  # Skip lines that do not end with a valid game result
                try:
                    prompt = prompt_generator(chess_format_dic, training_type)
                    if training_type == TrainingType.SUPERVISED_LEARNING:
                        input = tokenizer(prompt, return_tensors="pt", max_length=4000, padding="max_length", skip_special_tokens=True).to(device)
                    else:
                        input = tokenizer(prompt, return_tensors="pt", skip_special_tokens=True).to(device)
                    df.loc[len(df)] = [input["input_ids"].squeeze().numpy(), input["attention_mask"].squeeze().numpy()]
                    counter += 1
                except ValueError as e:
                    continue  # Skip games that do not meet the minimum move requirement
                if counter % 1000 == 0:
                    print(f"Processed {i} lines, current parquet file has {counter} data points.")
                if counter >= paraquet_limit:
                    parquet_filename = f"src\\chess\\paraquets\\{parquet_counter:06d}-{training_type.value}.parquet"
                    df.to_parquet(parquet_filename, compression="zstd", engine="pyarrow")
                    parquet_counter += 1
                    print(f"Successfully created a new Parquet file: '{parquet_filename}'")
                    counter = 0
                    df = pd.DataFrame(columns=['input_ids', 'attention_mask'])
                    if parquet_counter == max_paraquet_files_per_training_type:  # Limit to max_paraquet_files_per_training_type parquet files for this example
                        break
    return counter, parquet_counter
            

if __name__ == "__main__":
    # paraquet generation with 10000 data points in each parquet file
    sl_parequet_generated_toggle = False
    rl_parequet_generated_toggle = True
    parquet_sl_counter = 0
    parquet_rl_counter = 0
    paraquet_limit = 100000
    current_counter = 0
    for i in range(3,13):
        print(f"Processing file : lichess_db_standard_rated_2013-{i:02d}.pgn.txt.bz2")
        if not sl_parequet_generated_toggle:
            current_counter, sl_counter = open_bz2_file(f'src\chess\dataset\lichess_db_standard_rated_2013-{i:02d}.pgn.txt.bz2',current_counter, parquet_sl_counter, paraquet_limit, training_type=TrainingType.SUPERVISED_LEARNING)
            parquet_sl_counter += sl_counter
            print(f"Finished processing file : lichess_db_standard_rated_2013-{i:02d}.pgn.txt.bz2")
            current_counter = 0
            if parquet_sl_counter >= max_paraquet_files_per_training_type:
                sl_parequet_generated_toggle = True
                rl_parequet_generated_toggle = False
        if not rl_parequet_generated_toggle:
            current_counter, rl_counter = open_bz2_file(f'src\chess\dataset\lichess_db_standard_rated_2013-{i:02d}.pgn.txt.bz2',current_counter, parquet_rl_counter, paraquet_limit, training_type=TrainingType.REINFORCEMENT_LEARNING)
            parquet_rl_counter += rl_counter
            print(f"Finished processing file : lichess_db_standard_rated_2013-{i:02d}.pgn.txt.bz2")
            current_counter = 0
            if parquet_rl_counter >= max_paraquet_files_per_training_type:
                rl_parequet_generated_toggle = True
    
        