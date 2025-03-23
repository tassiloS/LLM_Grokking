from math import ceil
import torch
from tqdm import tqdm
import argparse

from data import get_data
from model import Transformer

def main(config):
    device = torch.device(config.device)

    # Get the data loaders
    train_loader, val_loader = get_data(
        config.operation,
        config.prime,
        config.training_fraction,
        config.batch_size
    )

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

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        global_step = train(model, train_loader, optimizer, scheduler, device, config.num_steps, global_step)
        evaluate(model, val_loader, device, epoch)
        if global_step >= config.num_steps:
            break

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
        print(metrics)

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
    print(metrics)


