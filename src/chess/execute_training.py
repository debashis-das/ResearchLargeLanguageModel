
from chess.rl_training import rl_execute
from chess.supervised_training import sft_execute

import argparse

def main():
    parser = argparse.ArgumentParser(description="Named arguments example")
    # 1. Standard named argument (with an optional short alias '-sft')
    parser.add_argument('-sft', '--supervised', type=str, default="yes", help="Is the training supervised? (yes/no)")
    # 2. Optional named argument with a default fallback value
    parser.add_argument('-rl', '--reinforcement', type=str, default="yes", help="Is the training reinforcement learning? (yes/no)")

    # Parse inputs
    args = parser.parse_args()

    # Access them using the long name (without the dashes)
    print(f"Is the training supervised? {args.supervised}")
    print(f"Is the training reinforcement learning? {args.reinforcement}")

    # Execute the SFT training
    if args.supervised.lower() == "yes":
        sft_execute()
    # Execute the RL training
    if args.reinforcement.lower() == "yes":
        rl_execute()

if __name__ == "__main__":
    main()
    
