# Environment

- OS: Windows
- Python: 3.11
- Environment: `.venv`
- PyTorch: use the installed CUDA-enabled build selected for the local NVIDIA driver
- GPU target: NVIDIA RTX 2050, 4 GB VRAM
- Default execution: FP32, batch size 8, CPU threads 2, DataLoader workers 0, sequential experiments

Verify the environment with:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
.\.venv\Scripts\python.exe test_setup.py
```

Training is never started automatically by opening VS Code or by running finalization/report commands.
