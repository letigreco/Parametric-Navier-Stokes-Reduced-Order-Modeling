"""
PODNN grid search.

Loads podnn_data.npz (exported from notebook) and searches over:
- Architecture (3-4 variants)
- Activation function (tanh, GELU)
- Weight decay (regularization)

For each configuration:
- Trains the network with constant LR + ReduceLROnPlateau
- Evaluates relative error vs FOM on 5 test mu
- Saves results to CSV

"""

import torch
import torch.nn as nn
import numpy as np
import time
import csv
import os
import sys


# CONFIGURATION

torch.set_default_dtype(torch.float32)

ARCHITECTURES = [
    {"name": "5x60",  "layers": [60]*5},
    {"name": "5x80",  "layers": [80]*5},
    {"name": "6x100", "layers": [100]*6},
    {"name": "7x60",  "layers": [60]*7},
]
ACTIVATIONS = ["tanh", "gelu"]
WEIGHT_DECAYS = [0.0, 1e-5, 1e-4]

MAX_EPOCHS = 80000
TOL = 1e-7
LR_INITIAL = 1e-3

# ReduceLROnPlateau parameters
PLATEAU_FACTOR = 0.5     # lr *= 0.5 when plateau detected
PLATEAU_PATIENCE = 3000  # wait this many epochs of no improvement before reducing
PLATEAU_MIN_LR = 1e-6    # don't go below this

OUTPUT_CSV = "podnn_gridsearch_results.csv"
LOG_FILE = "podnn_gridsearch_log.txt"
DATA_FILE = "podnn_data.npz"


# LOGGING

log_handle = open(LOG_FILE, "w")
def log(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_handle)
    log_handle.flush()


# LOAD DATA

if not os.path.exists(DATA_FILE):
    log(f"ERROR: {DATA_FILE} not found. Run export_podnn_data.py in the notebook first.")
    sys.exit(1)

data = np.load(DATA_FILE)
x_train_np = data["x_train"]
y_train_np = data["y_train"]
B_global = data["B_global"]
X_global = data["X_global"]
test_mus = data["test_mus"]
fom_test = data["fom_test"]
N_total = int(data["N_total"])
speed_n_dofs = int(data["speed_n_dofs"])
pressure_n_dofs = int(data["pressure_n_dofs"])
tot_dofs = int(data["tot_dofs"])

M = x_train_np.shape[0]
log(f"Loaded PODNN data:")
log(f"  M training samples: {M}")
log(f"  N reduced dim: {N_total}")
log(f"  N test mus: {len(test_mus)}")


# NORMALIZATION

mu_min = np.array([0.1, 1.0], dtype=np.float32)
mu_max = np.array([10.0, 3.0], dtype=np.float32)
x_train_norm = (x_train_np - mu_min) / (mu_max - mu_min)

y_mean = y_train_np.mean(axis=0)
y_std  = y_train_np.std(axis=0) + 1e-8
y_train_norm = (y_train_np - y_mean) / y_std

x_train_t = torch.tensor(x_train_norm)
y_train_t = torch.tensor(y_train_norm)


# NETWORK

class PODNN_Net(nn.Module):
    def __init__(self, layer_widths, activation):
        super().__init__()
        dims = [2] + list(layer_widths) + [N_total]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        self.activation = activation

    def forward(self, x):
        z = x
        for layer in self.layers[:-1]:
            z = self.activation(layer(z))
        return self.layers[-1](z)


# EVALUATION

def evaluate(net):
    """Compute relative L2 errors against FOM test set, in both Euclidean and X norm."""
    net.eval()
    errors_2 = []
    errors_X = []
    with torch.no_grad():
        for k, mu in enumerate(test_mus):
            mu_input = (mu - mu_min) / (mu_max - mu_min)
            y_norm = net(torch.tensor(mu_input).unsqueeze(0)).numpy()[0]
            alpha = y_norm * y_std + y_mean
            u_pred = B_global @ alpha
            u_ref = fom_test[k]

            # Euclidean
            err_2 = np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref)
            # X-norm
            e = u_pred - u_ref
            num = np.sqrt(abs(e @ (X_global @ e)))
            den = np.sqrt(abs(u_ref @ (X_global @ u_ref)))
            err_X = num / den

            errors_2.append(err_2)
            errors_X.append(err_X)
    net.train()
    return errors_2, errors_X


# TRAINING ONE CONFIG

def train_one(arch_dict, activation_name, weight_decay, run_id):
    log(f"\n{'='*60}")
    log(f"RUN {run_id}: arch={arch_dict['name']}, act={activation_name}, wd={weight_decay}")
    log(f"{'='*60}")

    if activation_name == "tanh":
        act = torch.tanh
    elif activation_name == "gelu":
        act = torch.nn.functional.gelu
    else:
        raise ValueError

    torch.manual_seed(31)
    net = PODNN_Net(arch_dict["layers"], act)
    n_params = sum(p.numel() for p in net.parameters())
    log(f"Parameters: {n_params}")

    optimizer = torch.optim.Adam(net.parameters(), lr=LR_INITIAL, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=PLATEAU_FACTOR,
        patience=PLATEAU_PATIENCE, min_lr=PLATEAU_MIN_LR)

    loss_fn = nn.MSELoss()

    t_start = time.time()
    last_lr = LR_INITIAL
    for epoch in range(1, MAX_EPOCHS + 1):
        optimizer.zero_grad()
        out = net(x_train_t)
        loss = loss_fn(out, y_train_t)

        if not torch.isfinite(loss):
            log(f"  DIVERGED at epoch {epoch}")
            return {
                "run_id": run_id, "arch": arch_dict["name"], "activation": activation_name,
                "weight_decay": weight_decay, "n_params": n_params, "epochs_done": epoch,
                "final_loss": float('nan'), "mean_err_2": float('nan'), "mean_err_X": float('nan'),
                "max_err_2": float('nan'), "max_err_X": float('nan'),
                "training_time_s": time.time() - t_start, "final_lr": float('nan'),
                "diverged": True,
            }

        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != last_lr:
            log(f"  epoch {epoch}: LR reduced to {current_lr:.2e}")
            last_lr = current_lr

        if epoch % 5000 == 0:
            log(f"  epoch {epoch:6d}  loss={loss.item():.3e}  lr={current_lr:.2e}")

        if loss.item() < TOL:
            log(f"  Tolerance {TOL} reached at epoch {epoch}")
            break

    t_train = time.time() - t_start
    final_loss = loss.item()

    errors_2, errors_X = evaluate(net)
    log(f"  Final loss: {final_loss:.3e}")
    log(f"  Errors Euclid: {[f'{e:.3e}' for e in errors_2]}")
    log(f"  Errors X-norm: {[f'{e:.3e}' for e in errors_X]}")
    log(f"  Mean err Euclid: {np.mean(errors_2):.3e},  X-norm: {np.mean(errors_X):.3e}")
    log(f"  Training time: {t_train:.1f}s")

    return {
        "run_id": run_id, "arch": arch_dict["name"], "activation": activation_name,
        "weight_decay": weight_decay, "n_params": n_params, "epochs_done": epoch,
        "final_loss": final_loss,
        "mean_err_2": float(np.mean(errors_2)), "mean_err_X": float(np.mean(errors_X)),
        "max_err_2": float(np.max(errors_2)), "max_err_X": float(np.max(errors_X)),
        "training_time_s": t_train, "final_lr": current_lr,
        "diverged": False,
    }


# MAIN

def main():
    log(f"PODNN Grid Search")
    log(f"Architectures: {[a['name'] for a in ARCHITECTURES]}")
    log(f"Activations:   {ACTIVATIONS}")
    log(f"Weight decays: {WEIGHT_DECAYS}")
    log(f"Max epochs:    {MAX_EPOCHS}")
    log(f"Scheduler:     ReduceLROnPlateau(factor={PLATEAU_FACTOR}, patience={PLATEAU_PATIENCE})")
    n_configs = len(ARCHITECTURES) * len(ACTIVATIONS) * len(WEIGHT_DECAYS)
    log(f"Total configs: {n_configs}")

    fieldnames = ["run_id", "arch", "activation", "weight_decay", "n_params",
                  "epochs_done", "final_loss", "final_lr",
                  "mean_err_2", "mean_err_X", "max_err_2", "max_err_X",
                  "training_time_s", "diverged"]
    csv_handle = open(OUTPUT_CSV, "w", newline="")
    writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
    writer.writeheader()

    t_global = time.time()
    run_id = 0
    all_results = []
    for arch in ARCHITECTURES:
        for act in ACTIVATIONS:
            for wd in WEIGHT_DECAYS:
                run_id += 1
                result = train_one(arch, act, wd, run_id)
                all_results.append(result)
                writer.writerow(result)
                csv_handle.flush()

    csv_handle.close()
    log(f"\n{'='*60}")
    log(f"Grid search complete in {(time.time() - t_global)/3600:.2f} hours")
    log(f"{'='*60}")

    valid = [r for r in all_results if not r["diverged"]]
    # Sort by mean X-norm error (most representative for fluid problems)
    valid.sort(key=lambda r: r["mean_err_X"])
    log(f"\nTOP 5 by X-norm error:")
    for i, r in enumerate(valid[:5]):
        log(f"  {i+1}. arch={r['arch']:6s}  act={r['activation']:4s}  "
            f"wd={r['weight_decay']:.0e}  "
            f"err_X={r['mean_err_X']:.3e}  err_2={r['mean_err_2']:.3e}  "
            f"time={r['training_time_s']:.0f}s")
    if len(valid) < len(all_results):
        log(f"\n{len(all_results) - len(valid)} configurations diverged.")

if __name__ == "__main__":
    main()