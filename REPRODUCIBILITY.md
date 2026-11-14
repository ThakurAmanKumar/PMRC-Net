# Reproducibility

```powershell
.\.venv\Scripts\python.exe test_setup.py
.\.venv\Scripts\python.exe stage2_validation.py --stage pmrc
.\.venv\Scripts\python.exe stage2_validation.py --stage baselines
.\.venv\Scripts\python.exe stage2_validation.py --stage ablations
.\.venv\Scripts\python.exe stage2_analysis.py
```

Run the commands one at a time. The PMRC command is the primary Stage 2 run. Results, fold assignments, configurations, checkpoints, and resource snapshots are written under `results/stage2_validation`.

## Finalization without retraining

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\finalize_project.py
.\.venv\Scripts\python.exe results\stage5_final_validation\stage5_validation.py --task reports
```

These commands read saved evidence and do not start training. The final package records dataset, configuration, and checkpoint hashes, five-seed OOF integrity, external-data screening, and known limitations. Do not claim independent external validation unless patient-level source independence is demonstrated and a frozen-model evaluation is separately approved.