"""PyTorch PatchTST prototype for the seven-step ATSQA workflow.

This is the practical version of the project idea:
1. compute quality scores q for each training sample;
2. learn quality-dimension weights w=softmax(alpha);
3. convert quality scores into sample weights;
4. run one differentiable virtual inner update;
5. update alpha on a meta-validation loss;
6. train PatchTST with stop-gradient sample weights;
7. compare against a uniform-weight PatchTST baseline.

Run with the user's environment:
    D:\\develop\\Anaconda3-5.2.0aaa\\envs\\library\\python.exe run_torch_patchtst_demo.py
"""

from __future__ import print_function

import argparse
import copy
import os
from collections import OrderedDict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

from atsqa_quality import quality_matrix
from models import PatchTST


QUALITY_NAMES = ["forecastability", "seasonality", "trend", "sparsity"]


def make_synthetic_mts(n_samples=240, seq_len=48, pred_len=1, n_vars=3, seed=7):
    rng = np.random.RandomState(seed)
    x = np.zeros((n_samples, seq_len, n_vars), dtype=np.float32)
    y = np.zeros((n_samples, pred_len, n_vars), dtype=np.float32)
    total_len = seq_len + pred_len

    for i in range(n_samples):
        t = np.arange(total_len, dtype=np.float32)
        kind = i % 4
        for d in range(n_vars):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            period = rng.choice([8, 12, 16, 24])
            amp = rng.uniform(0.6, 1.4)
            trend = rng.uniform(-0.025, 0.025) * t
            seasonal = amp * np.sin(2.0 * np.pi * t / period + phase)

            if kind == 0:
                series = seasonal + trend + rng.normal(0.0, 0.08, size=total_len)
            elif kind == 1:
                series = seasonal + trend + rng.normal(0.0, 0.35, size=total_len)
            elif kind == 2:
                series = trend * 3.0 + rng.normal(0.0, 0.12, size=total_len)
            else:
                series = seasonal + trend + rng.normal(0.0, 0.12, size=total_len)
                mask = rng.rand(total_len) < 0.72
                series[mask] = np.round(series[mask], 0)

            x[i, :, d] = series[:seq_len]
            y[i, :, d] = series[seq_len:]

    return x, y


def load_csv_windows(data_path, seq_len, pred_len, target=None, max_samples=None):
    """Load a Time-Series-Library style CSV into sliding windows.

    The common datasets use a first date column and numeric feature columns.
    If target is provided, only that column is predicted; otherwise all numeric
    columns are used as multivariate inputs and targets.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("CSV loading requires pandas in the active environment.")

    df = pd.read_csv(data_path)
    if target:
        if target not in df.columns:
            raise ValueError("target column '%s' not found in %s" % (target, data_path))
        values = df[[target]].values.astype(np.float32)
    else:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            raise ValueError("No numeric columns found in %s" % data_path)
        values = numeric.values.astype(np.float32)

    total = seq_len + pred_len
    windows = values.shape[0] - total + 1
    if windows <= 0:
        raise ValueError("CSV is too short for seq_len + pred_len.")
    if max_samples is not None:
        windows = min(windows, max_samples)

    x = np.zeros((windows, seq_len, values.shape[1]), dtype=np.float32)
    y = np.zeros((windows, pred_len, values.shape[1]), dtype=np.float32)
    for i in range(windows):
        x[i] = values[i:i + seq_len]
        y[i] = values[i + seq_len:i + total]
    return x, y


def split_data(x, y, train_ratio=0.65, val_ratio=0.2):
    n = x.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        x[:n_train], y[:n_train],
        x[n_train:n_train + n_val], y[n_train:n_train + n_val],
        x[n_train + n_val:], y[n_train + n_val:],
    )


def standardize(train_x, val_x, test_x):
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True) + 1e-6
    return (train_x - mean) / std, (val_x - mean) / std, (test_x - mean) / std


def batch_loss(model, x, y, params=None, reduction="mean"):
    pred = functional_call(model, params, (x,)) if params is not None else model(x)
    loss = F.mse_loss(pred, y, reduction="none").mean(dim=(1, 2))
    return loss.mean() if reduction == "mean" else loss


def current_params(model):
    return OrderedDict((name, param) for name, param in model.named_parameters())


def quality_sample_weights(q_batch, alpha, tau):
    dim_weights = torch.softmax(alpha, dim=0)
    quality_score = q_batch.matmul(dim_weights)
    sample_weights = torch.softmax(quality_score / tau, dim=0)
    return dim_weights, quality_score, sample_weights


def one_meta_step(model, alpha, train_x, train_y, train_q, val_x, val_y, args):
    params = current_params(model)

    _, _, sample_weights = quality_sample_weights(train_q, alpha, args.tau)
    train_losses = batch_loss(model, train_x, train_y, reduction="none")
    weighted_loss = (sample_weights * train_losses).sum()

    grads = torch.autograd.grad(
        weighted_loss,
        tuple(params.values()),
        create_graph=True,
        allow_unused=False,
    )
    theta_tilde = OrderedDict(
        (name, param - args.inner_lr * grad)
        for (name, param), grad in zip(params.items(), grads)
    )

    meta_loss = batch_loss(model, val_x, val_y, params=theta_tilde, reduction="mean")
    grad_alpha = torch.autograd.grad(meta_loss, alpha)[0]
    with torch.no_grad():
        alpha -= args.alpha_lr * grad_alpha
    return float(meta_loss.detach().cpu())


def train_weighted_epoch(model, alpha, loader, optimizer, device, args):
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch_x, batch_y, batch_q in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_q = batch_q.to(device)

        with torch.no_grad():
            _, _, weights = quality_sample_weights(batch_q, alpha, args.tau)
        losses = batch_loss(model, batch_x, batch_y, reduction="none")
        loss = (weights.detach() * losses).sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        total_loss += float(loss.detach().cpu()) * batch_x.size(0)
        total_count += batch_x.size(0)
    return total_loss / max(total_count, 1)


def train_uniform_epoch(model, loader, optimizer, device, args):
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch_x, batch_y, _ in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        loss = batch_loss(model, batch_x, batch_y, reduction="mean")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * batch_x.size(0)
        total_count += batch_x.size(0)
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, x, y, device):
    model.eval()
    return float(batch_loss(model, x.to(device), y.to(device), reduction="mean").cpu())


def build_model(args):
    return PatchTST(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=args.variables,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    if args.data_path:
        print("Loading CSV data from %s" % args.data_path)
        x, y = load_csv_windows(
            args.data_path,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            target=args.target,
            max_samples=args.max_samples,
        )
        args.variables = x.shape[-1]
    else:
        x, y = make_synthetic_mts(
            n_samples=args.samples,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            n_vars=args.variables,
            seed=args.seed,
        )
    train_x_raw, train_y, val_x_raw, val_y, test_x_raw, test_y = split_data(x, y)

    print("Step 1/7: computing quality vectors q=[%s]" % ", ".join(QUALITY_NAMES))
    q = quality_matrix(train_x_raw).astype(np.float32)
    print("quality mean:", dict(zip(QUALITY_NAMES, np.mean(q, axis=0).round(4))))

    train_x, val_x, test_x = standardize(train_x_raw, val_x_raw, test_x_raw)
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    val_x = torch.tensor(val_x, dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.float32)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)
    train_q = torch.tensor(q, dtype=torch.float32)

    dataset = TensorDataset(train_x, train_y, train_q)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    meta_loader = DataLoader(dataset, batch_size=args.meta_batch_size, shuffle=True, drop_last=True)
    meta_iter = iter(meta_loader)

    model = build_model(args).to(device)
    baseline = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)
    base_optimizer = torch.optim.Adam(baseline.parameters(), lr=args.model_lr)
    alpha = torch.zeros(4, dtype=torch.float32, device=device, requires_grad=True)

    print("\nRunning steps 2-6 with PatchTST for %d epochs..." % args.epochs)
    for epoch in range(1, args.epochs + 1):
        try:
            meta_batch = next(meta_iter)
        except StopIteration:
            meta_iter = iter(meta_loader)
            meta_batch = next(meta_iter)

        meta_x, meta_y, meta_q = [item.to(device) for item in meta_batch]
        meta_loss = one_meta_step(
            model, alpha, meta_x, meta_y, meta_q,
            val_x.to(device), val_y.to(device), args,
        )

        train_loss = train_weighted_epoch(model, alpha, loader, optimizer, device, args)
        base_train_loss = train_uniform_epoch(baseline, loader, base_optimizer, device, args)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            val_loss = evaluate(model, val_x, val_y, device)
            base_val = evaluate(baseline, val_x, val_y, device)
            dim_weights = torch.softmax(alpha.detach(), dim=0).cpu().numpy()
            weights_text = ", ".join(
                "%s=%.3f" % (name, value)
                for name, value in zip(QUALITY_NAMES, dim_weights)
            )
            print(
                "epoch %03d | train %.5f | val %.5f | base train %.5f | base val %.5f | meta %.5f | %s"
                % (epoch, train_loss, val_loss, base_train_loss, base_val, meta_loss, weights_text)
            )

    final_val = evaluate(model, val_x, val_y, device)
    final_test = evaluate(model, test_x, test_y, device)
    base_val = evaluate(baseline, val_x, val_y, device)
    base_test = evaluate(baseline, test_x, test_y, device)
    print("\nStep 7/7: feasibility check")
    print("uniform PatchTST:  val %.5f | test %.5f" % (base_val, base_test))
    print("adaptive ATSQA:    val %.5f | test %.5f" % (final_val, final_test))
    print("learned alpha:", alpha.detach().cpu().numpy().round(4))
    print(
        "learned quality weights:",
        dict(zip(QUALITY_NAMES, torch.softmax(alpha.detach(), dim=0).cpu().numpy().round(4))),
    )


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--variables", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--meta-batch-size", type=int, default=32)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--model-lr", type=float, default=0.001)
    parser.add_argument("--alpha-lr", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    return parser


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run(build_arg_parser().parse_args())
