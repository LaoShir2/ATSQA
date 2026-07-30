"""Runnable prototype for the seven-step ATSQA workflow.

This script validates the idea before a full deep-learning implementation:
1. compute four quality scores for each training sample;
2. generate quality-dimension weights with softmax(alpha);
3. convert quality scores into sample weights;
4. perform one weighted virtual inner update;
5. update alpha by validation loss through the virtual update;
6. train the predictor with the refreshed sample weights and stop-gradient;
7. report whether the learned weighting improves validation loss.

Run:
    python run_seven_step_demo.py
"""

from __future__ import division, print_function

import argparse
import os
import sys

import numpy as np

from atsqa_quality import quality_matrix


QUALITY_NAMES = ["forecastability", "seasonality", "trend", "sparsity"]


def softmax(x, temperature=1.0):
    z = np.asarray(x, dtype=np.float64) / float(temperature)
    z = z - np.max(z)
    ez = np.exp(z)
    return ez / (np.sum(ez) + 1e-12)


def make_synthetic_mts(n_samples=240, window=48, n_vars=3, seed=7):
    """Create data with mixed quality patterns for a one-step forecast task."""
    rng = np.random.RandomState(seed)
    x = np.zeros((n_samples, window, n_vars), dtype=np.float64)
    y = np.zeros((n_samples, n_vars), dtype=np.float64)

    for i in range(n_samples):
        t = np.arange(window + 1, dtype=np.float64)
        kind = i % 4
        for d in range(n_vars):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            period = rng.choice([8, 12, 16, 24])
            amp = rng.uniform(0.6, 1.4)
            trend = rng.uniform(-0.025, 0.025) * t
            seasonal = amp * np.sin(2.0 * np.pi * t / period + phase)

            if kind == 0:
                noise = rng.normal(0.0, 0.08, size=t.size)
                series = seasonal + trend + noise
            elif kind == 1:
                noise = rng.normal(0.0, 0.35, size=t.size)
                series = seasonal + trend + noise
            elif kind == 2:
                noise = rng.normal(0.0, 0.12, size=t.size)
                series = trend * 3.0 + noise
            else:
                base = seasonal + trend + rng.normal(0.0, 0.12, size=t.size)
                mask = rng.rand(t.size) < 0.72
                series = base
                series[mask] = np.round(series[mask], 0)

            x[i, :, d] = series[:-1]
            y[i, d] = series[-1]

    return x, y


def flatten_samples(x):
    return x.reshape((x.shape[0], x.shape[1] * x.shape[2]))


class LinearForecaster(object):
    """A small linear model with explicit per-sample gradients."""

    def __init__(self, n_features, n_outputs, seed=0):
        rng = np.random.RandomState(seed)
        self.w = rng.normal(0.0, 0.02, size=(n_features, n_outputs))
        self.b = np.zeros(n_outputs, dtype=np.float64)

    def copy_params(self):
        return self.w.copy(), self.b.copy()

    def set_params(self, params):
        self.w = params[0].copy()
        self.b = params[1].copy()

    def predict_with(self, x, params):
        w, b = params
        return np.dot(x, w) + b

    def predict(self, x):
        return self.predict_with(x, (self.w, self.b))

    def per_sample_loss_and_grads(self, x, y, params=None):
        if params is None:
            params = (self.w, self.b)
        pred = self.predict_with(x, params)
        err = pred - y
        losses = np.mean(err ** 2, axis=1)
        scale = 2.0 / y.shape[1]
        grad_w = np.einsum("ni,nj->nij", x, err) * scale
        grad_b = err * scale
        return losses, grad_w, grad_b

    def loss_and_grad(self, x, y, params=None):
        losses, grad_w, grad_b = self.per_sample_loss_and_grads(x, y, params)
        return float(np.mean(losses)), np.mean(grad_w, axis=0), np.mean(grad_b, axis=0)

    def apply_grad(self, grad_w, grad_b, lr):
        self.w -= lr * grad_w
        self.b -= lr * grad_b


def weighted_virtual_update(model, train_x, train_y, sample_weights, inner_lr):
    _, grad_w_i, grad_b_i = model.per_sample_loss_and_grads(train_x, train_y)
    grad_w = np.sum(sample_weights[:, None, None] * grad_w_i, axis=0)
    grad_b = np.sum(sample_weights[:, None] * grad_b_i, axis=0)
    theta = model.copy_params()
    theta_tilde = (theta[0] - inner_lr * grad_w, theta[1] - inner_lr * grad_b)
    return theta_tilde, grad_w_i, grad_b_i


def sample_weights_from_quality(q, alpha, tau):
    dim_weights = softmax(alpha)
    scores = np.dot(q, dim_weights)
    sample_weights = softmax(scores, temperature=tau)
    return dim_weights, scores, sample_weights


def meta_alpha_grad(q, dim_weights, scores, sample_weights, grad_w_i, grad_b_i,
                    val_grad_w, val_grad_b, inner_lr, tau):
    """Gradient of validation loss w.r.t. alpha through sample weights.

    The predictor parameters are treated as fixed for this one virtual update,
    which is the usual first-order approximation and is enough for a feasibility
    check of the quality-weight learning mechanism.
    """
    n, m = q.shape
    dz_dalpha = np.zeros((n, m), dtype=np.float64)
    for k in range(m):
        dz_dalpha[:, k] = dim_weights[k] * (q[:, k] - scores) / float(tau)

    # Softmax Jacobian over samples: da_i/dalpha_k = a_i(dz_i - sum_j a_j dz_j).
    centered = dz_dalpha - np.sum(sample_weights[:, None] * dz_dalpha, axis=0)[None, :]
    da_dalpha = sample_weights[:, None] * centered

    grad_alpha = np.zeros(m, dtype=np.float64)
    for k in range(m):
        dgw = np.sum(da_dalpha[:, k][:, None, None] * grad_w_i, axis=0)
        dgb = np.sum(da_dalpha[:, k][:, None] * grad_b_i, axis=0)
        grad_alpha[k] = -inner_lr * (np.sum(val_grad_w * dgw) + np.sum(val_grad_b * dgb))
    return grad_alpha


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
    mean = np.mean(train_x, axis=0, keepdims=True)
    std = np.std(train_x, axis=0, keepdims=True) + 1e-8
    return (train_x - mean) / std, (val_x - mean) / std, (test_x - mean) / std


def train_uniform_baseline(train_x, train_y, val_x, val_y, test_x, test_y, args):
    baseline = LinearForecaster(train_x.shape[1], train_y.shape[1], seed=args.seed)
    uniform = np.ones(train_x.shape[0], dtype=np.float64) / float(train_x.shape[0])
    best_val = None
    for _ in range(args.epochs):
        _, grad_w_i, grad_b_i = baseline.per_sample_loss_and_grads(train_x, train_y)
        grad_w = np.sum(uniform[:, None, None] * grad_w_i, axis=0)
        grad_b = np.sum(uniform[:, None] * grad_b_i, axis=0)
        baseline.apply_grad(grad_w, grad_b, args.model_lr)
        val_loss, _, _ = baseline.loss_and_grad(val_x, val_y)
        best_val = val_loss if best_val is None else min(best_val, val_loss)
    test_loss, _, _ = baseline.loss_and_grad(test_x, test_y)
    return best_val, test_loss


def run(args):
    np.set_printoptions(precision=4, suppress=True)
    x, y = make_synthetic_mts(
        n_samples=args.samples,
        window=args.window,
        n_vars=args.variables,
        seed=args.seed,
    )
    train_x_raw, train_y, val_x_raw, val_y, test_x_raw, test_y = split_data(x, y)

    print("Step 1/7: computing quality vectors q=[%s]" % ", ".join(QUALITY_NAMES))
    q = quality_matrix(train_x_raw)
    print("quality mean:", dict(zip(QUALITY_NAMES, np.mean(q, axis=0).round(4))))

    train_x, val_x, test_x = standardize(
        flatten_samples(train_x_raw),
        flatten_samples(val_x_raw),
        flatten_samples(test_x_raw),
    )

    model = LinearForecaster(train_x.shape[1], train_y.shape[1], seed=args.seed)
    alpha = np.zeros(4, dtype=np.float64)
    base_val, base_test = train_uniform_baseline(
        train_x, train_y, val_x, val_y, test_x, test_y, args
    )

    print("\nRunning steps 2-6 for %d epochs..." % args.epochs)
    for epoch in range(1, args.epochs + 1):
        # Step 2: quality-dimension weights.
        dim_weights, scores, sample_weights = sample_weights_from_quality(q, alpha, args.tau)

        # Step 3-4: sample weights and one virtual inner update.
        theta_tilde, grad_w_i, grad_b_i = weighted_virtual_update(
            model, train_x, train_y, sample_weights, args.inner_lr
        )

        # Step 5: update alpha from validation loss on theta_tilde.
        meta_loss, val_grad_w, val_grad_b = model.loss_and_grad(val_x, val_y, theta_tilde)
        grad_alpha = meta_alpha_grad(
            q, dim_weights, scores, sample_weights, grad_w_i, grad_b_i,
            val_grad_w, val_grad_b, args.inner_lr, args.tau
        )
        alpha -= args.alpha_lr * grad_alpha

        # Step 6: recompute weights after alpha update and train real model.
        dim_weights, _, sample_weights = sample_weights_from_quality(q, alpha, args.tau)
        _, grad_w_i, grad_b_i = model.per_sample_loss_and_grads(train_x, train_y)
        train_grad_w = np.sum(sample_weights[:, None, None] * grad_w_i, axis=0)
        train_grad_b = np.sum(sample_weights[:, None] * grad_b_i, axis=0)
        model.apply_grad(train_grad_w, train_grad_b, args.model_lr)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            train_loss, _, _ = model.loss_and_grad(train_x, train_y)
            val_loss, _, _ = model.loss_and_grad(val_x, val_y)
            weights_text = ", ".join(
                "%s=%.3f" % (name, value)
                for name, value in zip(QUALITY_NAMES, dim_weights)
            )
            print(
                "epoch %03d | train %.5f | val %.5f | meta %.5f | %s"
                % (epoch, train_loss, val_loss, meta_loss, weights_text)
            )

    # Step 7: final feasibility check.
    final_val, _, _ = model.loss_and_grad(val_x, val_y)
    test_loss, _, _ = model.loss_and_grad(test_x, test_y)
    print("\nStep 7/7: final test loss %.5f" % test_loss)
    print("uniform baseline: val %.5f | test %.5f" % (base_val, base_test))
    print("adaptive quality: val %.5f | test %.5f" % (final_val, test_loss))
    print("learned alpha:", alpha.round(4))
    print("learned quality weights:", dict(zip(QUALITY_NAMES, softmax(alpha).round(4))))
    return test_loss


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--variables", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--inner-lr", type=float, default=0.04)
    parser.add_argument("--model-lr", type=float, default=0.04)
    parser.add_argument("--alpha-lr", type=float, default=0.8)
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    return parser


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.exit(0 if run(build_arg_parser().parse_args()) >= 0.0 else 1)
