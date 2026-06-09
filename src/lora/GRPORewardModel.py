import json
import re

from torch import nn
import torch
from transformers import AutoTokenizer

from chess.chess_validator import ChessGame
from lora.LoRAFineTuning import LoRAFineTuning

class GRPORewardModel(nn.Module):

    def __init__(self, tokenizer: AutoTokenizer, model: LoRAFineTuning, grpo_batch: int, device="cpu", dtype=torch.float16, total_generation_length=400):
        super(GRPORewardModel, self).__init__()
        self.tokenizer = tokenizer
        self.grpo_batch = grpo_batch
        self.model = model
        self.device = device
        self.dtype = dtype
        self.total_generation_length = total_generation_length
        self.reward_moves = 5
        self.value_fn_moves = 10
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = 0.99

    
    def board_state(self, moves, chess_board: ChessGame, reward = 0.0, ignore_moves_till = 0):
        moves_clean = re.sub(r'\s*(1-0|0-1|1/2-1/2|\*)\s*$', '', moves.strip())
        # Match: move_number. white_move [black_move]
        pattern = r'(\d+)\.\s+(\S+)(?:\s+(?!\d+\.)(\S+))?'
        count_valid_moves = 0
        all_valid = True
        move_no = 0
        for m in re.finditer(pattern, moves_clean):
            move_no = int(m.group(1))
            if move_no <= ignore_moves_till:
                continue
            white   = m.group(2)
            black   = m.group(3)  # None if Black didn't play (resignation)
            if white:
                ok, _ = chess_board.push_san(white)
                if ok:
                    count_valid_moves += 1
                else:
                    print(f"Invalid move for white: {white}")
                    all_valid = False
                    break
            if black:
                ok, _ = chess_board.push_san(black)
                if ok:
                    count_valid_moves += 1
                else:
                    print(f"Invalid move for black: {black}")
                    all_valid = False
                    break
        if count_valid_moves == 0:
            reward -= 10.0
        else:
            reward += count_valid_moves * 0.5
        if all_valid:
            reward += 5.0
        else:
            move_no = move_no - 1  # Adjust move number if the last move was invalid
        return chess_board, reward, all_valid, move_no

    # reward for proper format of the output
    def chess_reward_function(self, output_with_prompt):
        game = ChessGame()
        # input extraction and create board state based on the input moves
        input_moves =  (output_with_prompt.rsplit("Generation Instructions:")[0].strip().rsplit("moves:")[-1].strip())
        # print(f"Input moves extracted for board state initialization: {input_moves}")
        game, _, _, move_no = self.board_state(input_moves, game)

        generation_moves = (output_with_prompt.rsplit("moves:")[-1].strip())
        # print(f"Generation moves extracted for reward calculation: {generation_moves}")
        play_as = (output_with_prompt.rsplit("play_as:")[-1].strip().split("moves:")[0].strip())
        reward = 0.0
        try:
            _, reward, all_valid, generation_move_no = self.board_state(generation_moves, game, reward, ignore_moves_till = move_no)
            # print(f"Reward after processing moves: {reward}, all_valid: {all_valid}, move_no: {move_no}")
            if all_valid:
                result = generation_moves.rsplit(" ")[-1]
                if play_as == "white" and "1-0" in generation_moves:
                    reward += 10.0
                elif play_as == "black" and "0-1" in generation_moves:
                    reward += 10.0
                elif play_as in ["white", "black"] and "1/2-1/2" in generation_moves:                            
                    reward += 5.0
        except Exception as e:
            print(f"An error occurred during move processing: {e}")
            reward -= 1.0
        if generation_move_no - move_no > 0:
            print(game.board_string())
        print(f"Generated new moves : {generation_move_no - move_no} : Reward : {reward}")
        return reward


    def extract_reward(self, tensor_per_generation: torch.Tensor):
        value_fn_str = self.tokenizer.decode(tensor_per_generation, skip_special_tokens=True)
        # print(f"Value function string :  {value_fn_str}")
        reward = self.chess_reward_function(value_fn_str)
        return reward
    
    # batch_size provided as input is always 1 
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        attention_mask = attention_mask.unsqueeze(0)
        x = x.unsqueeze(0)
        X = x.repeat_interleave(repeats=self.grpo_batch, dim=0)  # Repeat the input tensor for the batch size
        # print(f"Input shape : {X.shape}")
        output_tensor = self.model.generate(X, max_new_tokens=self.total_generation_length)
        # print(f"Output tensor shape: {output_tensor.shape}")
        advantage_batch = []
        nll_batch = [] 
        for idx, tensor_per_generation in enumerate(output_tensor):
            considered_tensor = tensor_per_generation
            # self.debug_logs(X, idx, tensor_per_generation)
            reward = self.extract_reward(considered_tensor)
        #     value_t_with_k_reward = self.extract_reward(tensor_per_generation, moves_to_consider=20)
        #     # print(f"Reward extracted: {reward} : Value function with k reward extracted: {value_t_with_k_reward}")
        #     value_t_reward = 0.0
            mask_addition = considered_tensor.shape[-1] - attention_mask.shape[-1]
            extra_mask = torch.ones(mask_addition, dtype=attention_mask.dtype, device=attention_mask.device).unsqueeze(0)
            training_mask = torch.cat([attention_mask, extra_mask], dim=-1)
        #     # print(f"Attention mask shape after concatenation : {attention_mask.shape} : {extra_mask.shape} : {training_mask.shape}")
            _, nll = self.model(considered_tensor.unsqueeze(0), attention_mask=training_mask)
            # advantage = reward +(value_t_with_k_reward - value_t_reward)
            advantage = reward
            nll_batch.append(nll)
            advantage_batch.append(torch.tensor(advantage, dtype=self.dtype, device=self.device))
        nll_batch = torch.stack(nll_batch)
        advantage_batch = torch.tanh(torch.stack(advantage_batch))
        print(f"Advantage : {[a.item() for a in advantage_batch]} : Loss : {[loss.item() for loss in nll_batch]}")
        loss_batch = nll_batch * self.gamma * advantage_batch
        print(f"Loss batch : {[loss.item() for loss in loss_batch]}")
        return loss_batch.mean()

    def debug_logs(self, X, idx, tensor_per_generation):
        print(f"Input      : {self.tokenizer.decode(X[idx], skip_special_tokens=True)}")
        print("-------------------------------------------------------------")
        print(f"Generation : {self.tokenizer.decode(tensor_per_generation[X.shape[-1]:], skip_special_tokens=True)}")
        print("-------------------------------------------------------------")


def board_state(moves, chess_board: ChessGame, reward = 0.0, ignore_moves_till = 0):
        moves_clean = re.sub(r'\s*(1-0|0-1|1/2-1/2|\*)\s*$', '', moves.strip())
        # Match: move_number. white_move [black_move]
        pattern = r'(\d+)\.\s+(\S+)(?:\s+(?!\d+\.)(\S+))?'
        count_valid_moves = 0
        all_valid = True
        move_no = 0
        for m in re.finditer(pattern, moves_clean):
            move_no = int(m.group(1))
            if move_no <= ignore_moves_till:
                continue
            white   = m.group(2)
            black   = m.group(3)  # None if Black didn't play (resignation)
            if white:
                ok, _ = chess_board.push_san(white)
                if ok:
                    count_valid_moves += 1
                else:
                    print(f"Invalid move for white: {white}")
                    all_valid = False
                    break
            if black:
                ok, _ = chess_board.push_san(black)
                if ok:
                    count_valid_moves += 1
                else:
                    print(f"Invalid move for black: {black}")
                    all_valid = False
                    break
        if count_valid_moves == 0:
            reward -= 10.0
        else:
            reward += count_valid_moves * 0.5
        if all_valid:
            reward += 5.0
        else:
            move_no = move_no - 1  # Adjust move number if the last move was invalid
        return chess_board, reward, all_valid, move_no

# reward for proper format of the output
def chess_reward_function(output_with_prompt, moves_to_consider=10):
    game = ChessGame()
    # input extraction and create board state based on the input moves
    input_moves =  (output_with_prompt.rsplit("Generation Instructions:")[0].strip().rsplit("moves:")[-1].strip())
    print(f"Input moves extracted for board state initialization: {input_moves}")
    game, _, _, move_no = board_state(input_moves, game)

    generation_moves = (output_with_prompt.rsplit("moves:")[-1].strip())
    print(f"Generation moves extracted for reward calculation: {generation_moves}")
    play_as = (output_with_prompt.rsplit("play_as:")[-1].strip().split("moves:")[0].strip())
    reward = 0.0
    try:
        _, reward, all_valid, generation_move_no = board_state(generation_moves, game, reward, ignore_moves_till = move_no)
        # print(f"Reward after processing moves: {reward}, all_valid: {all_valid}, move_no: {move_no}")
        if all_valid:
            result = generation_moves.rsplit(" ")[-1]
            if play_as == "white" and "1-0" in generation_moves:
                reward += 10.0
            elif play_as == "black" and "0-1" in generation_moves:
                reward += 10.0
            elif play_as in ["white", "black"] and "1/2-1/2" in generation_moves:                            
                reward += 5.0
    except Exception as e:
        print(f"An error occurred during move processing: {e}")
        reward -= 1.0
    print(f"Reward after processing input moves: {move_no} and play as {play_as}")
    print(game.board_string())
    print(f"Generated new moves : {generation_move_no - move_no} : Reward : {reward}")
    return reward


# if __name__ == "__main__":
#     generation = f"""

#     Chess is a two-player abstract strategy board game played on an 8×8 grid (64 squares). Each player controls 16 pieces: one king, one queen, two rooks, two bishops, two knights, and eight pawns. White moves first, and the goal is to checkmate the opponent's king (threaten it with inescapable capture).
#     Piece Movement

#     King (K): One square in any direction; can castle once per game.
#     Queen (Q): Any number of squares along a rank, file, or diagonal.
#     Rook (R): Any number of squares along a rank or file.
#     Bishop (B): Any number of squares diagonally.
#     Knight (N): An "L"-shape — two squares in one direction, one in another; leaps over other pieces.
#     Pawn: One square forward (or two on its first move); captures diagonally. Special moves: en passant and promotion.

#     Standard Algebraic Notation (SAN) — PRIMARY FOCUS
#     SAN is the universal standard for recording chess moves. Files are labeled a–h (left to right from White's perspective), and ranks are labeled 1–8 (bottom to top from White's perspective). Every square has a unique coordinate (e.g., e4, g7).
#     Piece symbols: K = King, Q = Queen, R = Rook, B = Bishop, N = Knight. Pawns have no letter prefix.
#     Basic move format:
#     [Piece][destination square]

#     Qg5 — Queen moves to g5
#     e4 — Pawn moves to e4
#     Nf3 — Knight moves to f3
#     Bc4 — Bishop moves to c4

#     Captures — insert x before the destination:
#     [Piece]x[destination square]

#     Bxf3 — Bishop captures on f3
#     exd5 — Pawn on the e-file captures on d5 (pawn captures always include the origin file)
#     Nxe5 — Knight captures on e5

#     Disambiguation — when two identical pieces can reach the same square:
#     [Piece][file or rank]x?[destination]

#     Ngf3 — Knight from the g-file moves to f3
#     R1e2 — Rook on rank 1 moves to e2
#     Qh4xe1 — Queen on h4 captures on e1 (both file and rank given when necessary)

#     Special moves:

#     Kingside castling : 0-0
#     Queenside castling : 0-0-0
#     Pawn promotion : e8=Q or e8Q (piece chosen after =)
#     Check : + suffix (e.g., Qd7+)
#     Checkmate : # suffix (e.g., Qxf7#)

#     Result formats:
#     1-0 : White wins or black resigns mid game due to a losing position
#     0-1 : Black wins or white resigns mid game due to a losing position
#     1/2-1/2 : Draw

#     Main Objective:

#     The main objective of chess is to checkmate your opponent's king. This means putting the opponent's king in a position where it is under attack (in "check") and
#     there is no legal move to escape the attack. The game can also end in a draw if neither player can force a checkmate, if both players agree to a draw,
#     or if certain conditions are met (e.g., stalemate, threefold repetition, fifty-move rule). Sometimes players may resign if they believe they are in a losing position,
#     which also ends the game.

#     Understanding these patterns is the main objective of chess and the key to mastering the game.

#     Example game in SAN format:
#     1. e4 Nf6 2. e5 Nd5 3. d4 d6 4. c4 Nb6 5. exd6 cxd6 6. Nf3 g6 7. a4 Bg7 8. a5 Nb6d7 9. Bd2 Nc6 10. Bc3 O-O 11. Be2 b6 12. axb6 Nxb6 13. d5 Ne5 14. Nxe5 dxe5 15. c5 Nxd5 16. Bf3 Bb7 17. O-O f5 18. Bxd5+ Qxd5 19. Qxd5+ Bxd5 20. Na3 Rfc8 21. Nb5 Rxc5 22. Na3 e4 23. Nc2 Bxc3 24. bxc3 Rxc3 25. Ne3 e6 26. Nxd5 exd5 27. Rac1 Rxc1 28. Rxc1 d4 29. h3 d3 30. Kf1 a5 31. Ke1 a4 32. Ra1 a3 33. Kd2 Ra4 34. Ke3 Kf7 35. f3 Ke6 36. f4 Kd5 37. g3 Kc4 38. g4 Kc3 39. gxf5 gxf5 40. Rc1+ Kb2 41. Rc5 a2 42. Rb5+ Kc2 43. Rc5+ Kd1 44. Rxf5 a1=Q 45. Kf2 Qc1 46. Kg3 Qe3+ 47. Kh4 Qf2+ 48. Kg5 1-0

#     In the above game both sides playes 48 moves and white won the game (1-0). The moves are in SAN format with move number and dot separator between white and black moves.

#     You are a chess player playing as white or black based on "play_as" key and try to defeat your opponent. The moves will be in SAN format.
#     Your task it to predict the next set of moves for both black and white till you win the game.
#     The moves will be provided in the following format:

#     play_as: "<tells you whether to play as white or black>",
#     moves: "Moves in SAN format with move number and dot separator between white and black moves. For example : 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6",


#         Input :
#         play_as: white
#         moves: 1. d4 c5 2. d5 Qb6 3. Nc3 e6 4. dxe6 fxe6 5. Nf3 c4 6. Rb1 Nc6 7. Be3 Qc7 8. Nd4 Bd6 9. g3 Nge7 10. Nxc6 bxc6 11. Bg2 Bb7 12. f3 O-O-O

#         Generation Instructions:
#         Output should be generated in the same input format after the moves provided as input

#         play_as: white
#         moves: 1. d4 c5 2. d5 Qb6 3. Nc3 e6 4. dxe6 fxe6 5. Nf3 c4 6. Rb1 Nc6 7. Be3 Qc7 8. Nd4 Bd6 9. g3 Nge7 10. Nxc6 bxc6 11. Bg2 Bb7 12. f3 O-O-O



#     Output:
#     play_as: white
#     moves: 1. d4 c5 2. d5 Qb6 3. Nc3 e6 4. dxe6 fxe6 5. Nf3 c4 6. Rb1 Nc6 7. Be3 Qc7 8. Nd4 Bd6 9. g3 Nge7 10. Nxc6 bxc6 11. Bg2 Bb7 12. f3 O-O-O 13. Nf3 c5 14. d5 Qb6 15. Nc3 e6 16. dxe6 fxe6 17. Nf3 c4 18. Rb1 Nc6 19. Be3 Qc7 20. Nd4 Bd6 21. g3 Nge7 22. Nxc6 bxc6 23. Bg2 Bb7 24. f3 O-O-O 25. Nf3 c5 26. d5 Qb6 27. Nc3 e6 28. dxe6 fxe6 29. Nf3 c4 30. Rb1 Nc6 31. Be3 Qc7 32. Nd4 Bd6 33
# """
#     chess_reward_function(generation)
            
            

        




        