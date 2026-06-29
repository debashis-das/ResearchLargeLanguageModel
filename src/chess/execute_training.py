
from chess.rl_training import rl_execute
from chess.supervised_training import sft_execute


if __name__ == "__main__":
    # Execute the SFT training
    sft_execute()
    # Execute the RL training
    rl_execute()
