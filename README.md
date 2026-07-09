# TODO
Fix Nan issue
Rope Embedding check in changes
Check feasibility : Triton change the hidden_dimension (if possible)
Linear : Triton Implmentation

# ResearchLargeLanguageModel
Research on building LLMs

# Distributed kernel
torchrun --nproc-per-node=4 src/app/distkernel.py

## Run

python -m app.main

## Setup

**Windows (PowerShell) — activate venv:**
```powershell
.\make-dev-venv.ps1
Invoke-Expression -Command "$(Get-Content -Path .\.venv-win)\Scripts\Activate.ps1"
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install triton --upgrade

# for sm_120
pip uninstall torch torchvision torchaudio -y

pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128

--------------------------------------------------------------------

pip uninstall torch -y

pip install torch --index-url https://download.pytorch.org/whl/cu128

# verify using 
import torch

print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())  # should be (12, 0)
print(torch.version.cuda)                  # should be 12.8+

cd /home
mkdir model
cd model
hf download Qwen/Qwen3-0.6B-Base --local-dir .

#optional
pip install --upgrade scipy
# run for AMD
pip install "numpy>=1.17.3,<1.25.0"

nohup python -u src/chess/rl_training.py > log.txt 2>&1 &

Machine : RTX 3090

CUDA_LAUNCH_BLOCKING=1 python src/chess/rl_training.py
batch : 8
tokens : 4k–5k
embedding : 8k

# TODO
Checkpointing
# Integrate
normal distribution


# PyTorch pipeline API (Multi GPU) --> Not required as context length based sharding

from torch.distributed.pipeline.sync import Pipe

model = Pipe(model, chunks=4)

## Automatically:

splits batch
pipelines execution
overlaps compute

## Used in real systems
DeepSpeed
Megatron-LM

## Important constraint
Pipeline parallelism only helps when:
    - layers are on DIFFERENT GPUs


