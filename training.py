from math import ceil
import torch
from tqdm import tqdm
import argparse

from data import get_data
from model import Transformer

import matplotlib.pyplot as plt
import os

def main(config):
    device = torch.device(config.device)

    # Get the data loaders
    train_loader, val_loader = get_data(
        config.operation,
        config.prime,
        config.training_fraction,
        config.batch_size
    )
    
    # Print an example data sample before training
    example_inputs, example_labels = next(iter(train_loader))
    # Inputs are of the form [x, op_token, y, eq_token]
    example_x = example_inputs[0, 0].item()
    example_y = example_inputs[0, 2].item()
    example_result = example_labels[0].item()
    print(f'operation: "{config.operation}", example: x={example_x}, y={example_y}, {config.operation} = {example_result}')

    # Initialize the model
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

    num_epochs = ceil(config.num_steps / len(train_loader))
    global_step = 0
    val_accuracies = []  # To store validation accuracy per epoch

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        global_step = train(model, train_loader, optimizer, scheduler, device, config.num_steps, global_step)
        metrics_val = evaluate(model, val_loader, device, epoch)
        val_accuracies.append(metrics_val["validation/accuracy"])



        if global_step >= config.num_steps:
            break

    # After training, plot the validation accuracy over epochs
    plot_accuracy(val_accuracies, config)

def train(model, train_loader, optimizer, scheduler, device, num_steps, global_step):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    for batch in train_loader:
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch

        optimizer.zero_grad()
        output = model(inputs)[-1, :, :]
        loss = criterion(output, labels)
        acc = (torch.argmax(output, dim=1) == labels).sum() / len(labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        global_step += 1
        metrics = {
            "training/accuracy": acc.item(),
            "training/loss": loss.item(),
            "step": global_step
        }

        if global_step >= num_steps:
            break
    return global_step

def evaluate(model, val_loader, device, epoch):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    total_loss = 0.0
    total_samples = 0

    for batch in val_loader:
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch

        with torch.no_grad():
            output = model(inputs)[-1, :, :]
            predictions = torch.argmax(output, dim=1)
            correct += (predictions == labels).sum().item()
            batch_loss = criterion(output, labels).item()
            total_loss += batch_loss * len(labels)
            total_samples += len(labels)

    acc = correct / total_samples
    avg_loss = total_loss / total_samples
    metrics = {
        "validation/accuracy": acc,
        "validation/loss": avg_loss,
        "epoch": epoch
    }
    return metrics  # Return the metrics so you can track accuracy



def plot_accuracy(accuracies, config):
    # Ensure the results folder exists
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(accuracies)), accuracies, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Accuracy over Epochs\nOperation: {config.operation}, Prime: {config.prime}')
    plt.grid(True)
    
    # Save the plot with a filename that includes the parameters
    save_path = os.path.join('results', f"accuracy_plot_{config.operation}_{config.prime}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Accuracy plot saved to {save_path}")