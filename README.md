# Adaptive Time Series Quality Assessment Prototype

This folder contains a runnable, dependency-light prototype for the seven-step
workflow in the project document.

## Run

```bash
python run_seven_step_demo.py
```

For the PyTorch + PatchTST version, use the `library` conda environment:

```powershell
D:\develop\Anaconda3-5.2.0aaa\envs\library\python.exe run_torch_patchtst_demo.py
```

To run on a Time-Series-Library style CSV dataset:

```powershell
D:\develop\Anaconda3-5.2.0aaa\envs\library\python.exe run_torch_patchtst_demo.py --data-path path\to\ETTh1.csv --target OT --seq-len 96 --pred-len 24
```

Omit `--target` for multivariate forecasting over all numeric columns.

The original `run_seven_step_demo.py` uses only `numpy`, so it remains a
fallback baseline. `run_torch_patchtst_demo.py` is the main version: it creates
a synthetic multivariate time-series forecasting task, computes the four
quality scores from Figure 2, learns quality-dimension weights through a
one-step differentiable meta-validation update, then trains a compact PatchTST
forecaster with refreshed stop-gradient sample weights.

## Files

- `atsqa_quality.py`: quality scores:
  - forecastability: `1 - spectral_entropy / log(N_f)`
  - seasonality strength: `1 - Var(R) / Var(S + R)` using an FFT period estimate
  - trend strength: `min(1, abs(beta_hat) * T)` after min-max normalization
  - sparsity: `1 - N_unique(X) / T`
- `run_seven_step_demo.py`: complete seven-step executable experiment.
- `models/patchtst.py`: compact PatchTST-style forecaster with input shape
  `[batch, seq_len, channels]` and output shape `[batch, pred_len, channels]`.
- `run_torch_patchtst_demo.py`: PyTorch implementation of the seven-step
  workflow with PatchTST and a uniform-weight baseline.

## Notes

The PyTorch version differentiates through the quality-generated sample weights
and one virtual PatchTST update using `torch.autograd.grad(create_graph=True)`.
During the real model update, sample weights are detached, matching the
Stop-Gradient requirement in the project workflow.

The PatchTST module is self-contained but keeps the same forecasting tensor
convention used by THUML Time-Series-Library. If the full library is later
vendored into this project, `build_model()` in `run_torch_patchtst_demo.py` is
the intended replacement point.
