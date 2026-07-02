import bz2
import enum
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

max_paraquet_files_per_training_type = 4
# model_path = "/home/model"
model_path = "C:\\Users\\DebashisDas\\personal\\models\\Qwen"
df = pd.DataFrame(columns=['input_ids', 'attention_mask'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path)

def prompt_generator(chess_json, training_type):
    # move extraction logic for supervised learning
    def parse_moves(moves, play_as, num_moves_to_parse):
        """Parse PGN into list of (move_no, white, black) tuples."""
        # Remove result at end
        moves_clean = re.sub(r'\s*(1-0|0-1|1/2-1/2|\*)\s*$', '', moves.strip())
        # Match: move_number. white_move [black_move]
        pattern = r'(\d+)\.\s+(\S+)(?:\s+(?!\d+\.)(\S+))?'
        extracted_moves = ""
        generate_moves = ""
        for m in re.finditer(pattern, moves_clean):
            move_no = int(m.group(1))
            white   = m.group(2)
            black   = m.group(3)  # None if Black didn't play (resignation)
            if move_no <= num_moves_to_parse:
                extracted_moves += f"{move_no}. {white} {black if black else ''} "
            if play_as == "black" and move_no == num_moves_to_parse+1:
                extracted_moves += f"{move_no}. {white} "
                generate_moves += f"{black if black else ''} "
            if play_as == "white" and move_no > num_moves_to_parse:
                generate_moves += f"{move_no}. {white} {black if black else ''} "
            if play_as == "black" and move_no > num_moves_to_parse+1:
                generate_moves += f"{move_no}. {white} {black if black else ''} "
        return extracted_moves, generate_moves
    # Generate the prompt based on the input JSON
    move_numbers = re.findall(r'(\d+)\.', chess_json["moves"])
    num_moves = int(move_numbers[-1])
    if num_moves < 5:  # If the game has less than 10 moves, parse all moves
        raise ValueError(f"Game has only {num_moves} moves, which is less than the minimum required 10 moves for training.")
    num_moves_to_parse = random.randint((num_moves // 4), (num_moves // 4) * 3)  # Randomly choose to parse between 25% and 75% of the moves
    extracted_moves, generate_moves = parse_moves(chess_json["moves"], chess_json["play_as"], num_moves_to_parse=num_moves_to_parse)
    if training_type == TrainingType.SUPERVISED_LEARNING:
        prompt = f"""
        <system>
        You are a chess engine. You will be given the side you are playing and the move history so far. Continue the game from the current position.

        OUTPUT FORMAT (strict — required for automated parsing):
        - Output ONLY the continuation moves in Standard Algebraic Notation (SAN).
        - Format: "N. white_move black_move N+1. white_move black_move ..." with a single space between tokens, no newlines, no leading/trailing whitespace.
        - If it is Black's turn on the first move you output, start with "black_move N+1. white_move".
        - Do not repeat any moves from the given history.
        - Use standard SAN characters only (piece letters, files a-h, ranks 1-8, x for capture, = for promotion, O-O/O-O-O for castling). Include + for check and # for checkmate only when standard SAN requires them for disambiguation — otherwise omit.
        - Do not output result markers (1-0, 0-1, 1/2-1/2), the word "checkmate," "resigns," commentary, evaluations, or any text besides the move list.
        - Stop generating immediately once checkmate is delivered or you have produced the requested number of moves — do not add anything after the final move token.

        LEGALITY:
        - Every move must be strictly legal in the current position. Never output a move that does not exist on the board or violates the rules.
        - Track the board state implicitly from the full move history before choosing each move.

        PLAY STRENGTH:
        - At each turn, select the move you judge strongest given material balance, king safety, piece activity, and tactical threats.
        - Do not hedge between candidate moves or explain reasoning — commit to one move per ply.

        <user>
        play_as: {chess_json["play_as"]}
        moves: "{extracted_moves}"

        Task:
        - Continue the game
        - Play optimally
        - End only at checkmate or resignation
        - Output only moves

        <assistant>
        {generate_moves}
        """
    if training_type == TrainingType.REINFORCEMENT_LEARNING:
        prompt = f"""
        <system>
        You are a chess engine. You will be given the side you are playing and the move history so far. Continue the game from the current position.

        OUTPUT FORMAT (strict — required for automated parsing):
        - Output ONLY the continuation moves in Standard Algebraic Notation (SAN).
        - Format: "N. white_move black_move N+1. white_move black_move ..." with a single space between tokens, no newlines, no leading/trailing whitespace.
        - If it is Black's turn on the first move you output, start with "black_move N+1. white_move".
        - Do not repeat any moves from the given history.
        - Use standard SAN characters only (piece letters, files a-h, ranks 1-8, x for capture, = for promotion, O-O/O-O-O for castling). Include + for check and # for checkmate only when standard SAN requires them for disambiguation — otherwise omit.
        - Do not output result markers (1-0, 0-1, 1/2-1/2), the word "checkmate," "resigns," commentary, evaluations, or any text besides the move list.
        - Stop generating immediately once checkmate is delivered or you have produced the requested number of moves — do not add anything after the final move token.

        LEGALITY:
        - Every move must be strictly legal in the current position. Never output a move that does not exist on the board or violates the rules.
        - Track the board state implicitly from the full move history before choosing each move.

        PLAY STRENGTH:
        - At each turn, select the move you judge strongest given material balance, king safety, piece activity, and tactical threats.
        - Do not hedge between candidate moves or explain reasoning — commit to one move per ply.

        <user>
        play_as: {chess_json["play_as"]}
        moves: "{extracted_moves}"

        Task:
        - Continue the game
        - Play optimally
        - End only at checkmate or resignation
        - Output only moves

        <assistant>"""
    return prompt

def open_bz2_file(file_path, counter, parquet_counter, paraquet_limit, training_type):
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
    paraquet_limit = 5000
    current_counter = 0
    for i in range(3,13):
        print(f"Processing file : lichess_db_standard_rated_2013-{i:02d}.pgn.txt.bz2")
        if not sl_parequet_generated_toggle:
            current_counter, sl_counter = open_bz2_file(f'src\\chess\\dataset\\lichess_db_standard_rated_2013-{i:02d}.pgn.txt.bz2',current_counter, parquet_sl_counter, paraquet_limit, training_type=TrainingType.SUPERVISED_LEARNING)
            parquet_sl_counter += sl_counter
            print(f"Finished processing file : lichess_db_standard_rated_2013-{i:02d}.pgn.txt.bz2")
            current_counter = 0
            if parquet_sl_counter >= max_paraquet_files_per_training_type:
                sl_parequet_generated_toggle = True
                rl_parequet_generated_toggle = False
                df = df[0:0]  # Clear the DataFrame to free up memory before starting RL parquet generation
        if not rl_parequet_generated_toggle:
            current_counter, rl_counter = open_bz2_file(f'src\\chess\\dataset\\lichess_db_standard_rated_2013-{i:02d}.pgn.txt.bz2',current_counter, parquet_rl_counter, paraquet_limit, training_type=TrainingType.REINFORCEMENT_LEARNING)
            parquet_rl_counter += rl_counter
            print(f"Finished processing file : lichess_db_standard_rated_2013-{i:02d}.pgn.txt.bz2")
            current_counter = 0
            if parquet_rl_counter >= max_paraquet_files_per_training_type:
                rl_parequet_generated_toggle = True
    
        