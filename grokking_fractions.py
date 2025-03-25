#!/usr/bin/env python3
import numpy as np
import torch
import argparse
import os
import matplotlib.pyplot as plt
from math import ceil
from tqdm import tqdm
import time
from datetime import datetime

# Import your data and model functions.
from data import get_data
from model import Transformer
# Import training and evaluation functions from your training module.
from training import train, evaluate

def run_training_for_fraction(config, max_steps):
    """
    Trains a fresh model using the specified training_fraction (in config) until
    the validation accuracy exceeds 0.99 or max_steps gradient steps are reached.
    Returns the number of gradient steps required for grokking, or None if not reached,
    along with the elapsed training time in seconds.
    """
    device = torch.device(config.device)
    
    # Get data loaders with the given training fraction.
    train_loader, val_loader = get_data(
        config.operation,
        config.prime,
        config.training_fraction,
        config.batch_size
    )
    
    # Initialize a new model.
    model = Transformer(
        num_layers=config.num_layers,
        dim_model=config.dim_model,
        num_heads=config.num_heads,
        num_tokens=config.prime + 2,
        seq_len=5
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=9
    )
    
    global_step = 0
    start_time = time.time()
    # Run training until either we exceed max_steps or validation accuracy > 0.99 is achieved.
    while global_step < max_steps:
        global_step, _ = train(model, train_loader, optimizer, scheduler, device, max_steps, global_step)
        metrics_val = evaluate(model, val_loader, device, epoch=0)
        val_acc = metrics_val["validation/accuracy"]
        if val_acc >= 0.99:
            elapsed = time.time() - start_time
            return global_step, elapsed
    return None, time.time() - start_time

def main():
    default_device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", type=str, default="x+y")
    parser.add_argument("--prime", type=int, default=59)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1)
    # Set a maximal number of gradient steps (e.g. 10000) for each experiment.
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--device", type=str, default=default_device)
    args = parser.parse_args()
    
    # Generate fractions from 0.1 to 0.9 in 0.05 increments.
    fractions = np.arange(0.1, 0.9 + 0.0001, 0.05)
    grok_steps = []  # Stores gradient steps for grokking per fraction.
    grok_times = []  # Stores elapsed times (in seconds) per fraction.
    
    print("Starting grokking experiments for operation x+y ...")
    total_experiments = len(fractions)
    start_overall = time.time()
    for idx, frac in enumerate(fractions):
        args.training_fraction = float(frac)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{current_time}] Experiment {idx+1}/{total_experiments}: Training fraction = {frac:.3f}")
        
        steps, elapsed = run_training_for_fraction(args, args.num_steps)
        if steps is None:
            print(f"  Did not grok (val acc > 0.99) within {args.num_steps} gradient steps. Elapsed time: {elapsed:.1f}s")
            grok_steps.append(np.nan)
            grok_times.append(np.nan)
        else:
            print(f"  Grokking achieved in {steps} gradient steps. Elapsed time: {elapsed:.1f}s")
            grok_steps.append(steps)
            grok_times.append(elapsed)
        
        # Print an ETA for the remaining experiments.
        elapsed_overall = time.time() - start_overall
        experiments_done = idx + 1
        eta = (elapsed_overall / experiments_done) * (total_experiments - experiments_done)
        print(f"  Estimated time remaining: {eta/60:.1f} minutes")
    
    # Plot the training fraction vs. the number of gradient steps required for grokking.
    plt.figure(figsize=(8, 6))
    plt.plot(fractions, grok_steps, marker='o', linestyle='-')
    plt.xlabel("Training Data Fraction")
    plt.ylabel("Gradient Steps to Grok (Val Acc > 0.99)")
    plt.title(f"Grokking Time (Gradient Steps) vs Training Data Fraction\nOperation: x+y, Prime: {args.prime}")
    plt.grid(True)
    os.makedirs("results", exist_ok=True)
    steps_save_path = os.path.join("results", "grokking_time_vs_fraction.png")
    plt.savefig(steps_save_path)
    plt.close()
    print(f"\nFinal gradient steps plot saved to {steps_save_path}")

    # Plot the training fraction vs. elapsed time (in seconds).
    plt.figure(figsize=(8, 6))
    plt.plot(fractions, grok_times, marker='o', linestyle='-')
    plt.xlabel("Training Data Fraction")
    plt.ylabel("Time to Grok (seconds)")
    plt.title(f"Grokking Time (Seconds) vs Training Data Fraction\nOperation: x+y, Prime: {args.prime}")
    plt.grid(True)
    time_save_path = os.path.join("results", "grokking_time_seconds_vs_fraction.png")
    plt.savefig(time_save_path)
    plt.close()
    print(f"Final elapsed time plot saved to {time_save_path}")

if __name__ == "__main__":
    main()
