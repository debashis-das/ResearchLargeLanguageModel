# ResearchLargeLanguageModel
Research on building LLMs

# Distributed kernel
torchrun --nproc-per-node=4 src/app/distkernel.py

## Run

python -m app.main

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

