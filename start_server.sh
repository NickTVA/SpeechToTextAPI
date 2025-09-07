#!/bin/bash

# Set up CUDA library paths for PyTorch
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PATH="$SCRIPT_DIR/.venv1"

# Add all NVIDIA library paths
export LD_LIBRARY_PATH="$VENV_PATH/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_PATH/lib/python3.11/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_PATH/lib/python3.11/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_PATH/lib/python3.11/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_PATH/lib/python3.11/site-packages/nvidia/curand/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_PATH/lib/python3.11/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_PATH/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_PATH/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_PATH/lib/python3.11/site-packages/nvidia/cuda_cupti/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_PATH/lib/python3.11/site-packages/nvidia/nvtx/lib:$LD_LIBRARY_PATH"

# Start the Speech-to-Text API server on port 8001
uvicorn main:app --reload --host 0.0.0.0 --port 8001