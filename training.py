from math import ceil
import torch
from tqdm import tqdm
import argparse

from data import check_abelian
from data import get_data
from model import Transformer

import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA  # Alternatively, use TSNE from sklearn.manifold
import numpy as np

def main(config):
    device = torch.device(config.device)

    # Get the data loaders
    train_loader, val_loader = get_data(
        config.operation,
        config.prime,
        config.training_fraction,
        config.batch_size
    )
    # Check if the dataset forms an abelian group (if applicable)
    if config.operation == "abelian":
        print(f"The dataset forms an abelian group: {check_abelian(get_data, config.prime, config.training_fraction, config.batch_size)}")
    
    # Print an example data sample before training
    example_inputs, example_labels = next(iter(train_loader))
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
    train_accuracies = []
    val_accuracies = []  # To store validation accuracy per epoch

    """
    next_snapshot_step = 500

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        # Train for one epoch.
        global_step, train_acc = train(model, train_loader, optimizer, scheduler, device, config.num_steps, global_step)
        # Evaluate on the validation set.
        metrics_val = evaluate(model, val_loader, device, epoch)
        val_acc = metrics_val["validation/accuracy"]
        
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Save embedding snapshot every 100 gradient steps.
        while global_step >= next_snapshot_step:
            save_embedding_snapshot(model, config, snapshot_idx=next_snapshot_step, method="PCA", val_acc=val_acc)
            next_snapshot_step += 500

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Training accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}")

        if global_step >= config.num_steps:
            break
    """

    # After training, plot the training and validation accuracy over epochs.
    plot_accuracy(train_accuracies, val_accuracies, config)
    # Optionally, also save a final embedding plot.
    plot_embeddings(model, config, val_acc=val_accuracies[-1] if val_accuracies else None, method="PCA")

def train(model, train_loader, optimizer, scheduler, device, num_steps, global_step):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_correct = 0
    total_samples = 0

    for batch in train_loader:
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch

        optimizer.zero_grad()
        output = model(inputs)[-1, :, :]
        loss = criterion(output, labels)
        preds = torch.argmax(output, dim=1)
        correct_batch = (preds == labels).sum().item()
        total_correct += correct_batch
        total_samples += len(labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        global_step += 1
        if global_step >= num_steps:
            break
    epoch_training_accuracy = total_correct / total_samples if total_samples > 0 else 0
    return global_step, epoch_training_accuracy

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
    return metrics

def plot_accuracy(train_accuracies, val_accuracies, config, smoothing_factor=0.9):
    def smooth_curve(points, factor):
        smoothed = []
        for point in points:
            if smoothed:
                smoothed.append(smoothed[-1] * factor + point * (1 - factor))
            else:
                smoothed.append(point)
        return smoothed

    smoothed_train = smooth_curve(train_accuracies, smoothing_factor)
    smoothed_val = smooth_curve(val_accuracies, smoothing_factor)

    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    steps = range(len(smoothed_train))
    plt.plot(steps, smoothed_train, linestyle='-', color='red', label="Training Accuracy")
    plt.plot(steps, smoothed_val, linestyle='-', color='green', label="Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy over Epochs\nOperation: {config.operation}, Prime: {config.prime}, Training Fraction: {config.training_fraction}')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join('results', f"accuracy_plot_{config.operation.replace('/', '_mod_div_').replace('//', '_euc_div_')}_{config.prime}_{config.training_fraction}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Accuracy plot saved to {save_path}")

def plot_embeddings(model, config, val_acc=None, method="TSNE"):
    """
    Extracts the learned embeddings from the model's embedding layer,
    reduces them to 2D using PCA or t-SNE (specified by the 'method' argument),
    and plots the numbers (0 to prime-1) as text labels in the 2D space with gradually changing colors.
    If val_acc is provided, it is displayed in the title.
    """
    # Try to extract the embedding matrix.
    if hasattr(model, 'token_embeddings'):
        embeddings = model.token_embeddings.weight.detach().cpu()
    elif hasattr(model, 'embedding'):
        embeddings = model.embedding.weight.detach().cpu()
    else:
        raise ValueError("The model does not have an accessible embedding layer.")

    # We only want to analyze the numeric tokens (0 to prime-1)
    numeric_embeddings = embeddings[:config.prime]  # shape: [prime, dim_model]

    # Choose dimensionality reduction method.
    if method.upper() == "PCA":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        embedding_2d = reducer.fit_transform(numeric_embeddings.numpy())
        method_name = "PCA"
    elif method.upper() in ["TSNE", "T-SNE"]:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(numeric_embeddings.numpy())
        method_name = "t-SNE"
    else:
        raise ValueError("Invalid method. Choose 'PCA' or 't-SNE'.")

    # Determine axis limits based on data range, adding a margin.
    x_min, y_min = embedding_2d.min(axis=0)
    x_max, y_max = embedding_2d.max(axis=0)
    margin_x = (x_max - x_min) * 0.1
    margin_y = (y_max - y_min) * 0.1

    # Set up the plot.
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("viridis")
    for i, (x, y) in enumerate(embedding_2d):
        color = cmap(i / (config.prime - 1) if config.prime > 1 else 0)
        plt.text(x, y, str(i), fontsize=12, color=color, weight='bold')
    plt.xlabel(f"{method_name} Component 1")
    plt.ylabel(f"{method_name} Component 2")
    
    # Append validation accuracy info if provided.
    if val_acc is not None:
        title_extra = f", Validation Accuracy: {val_acc:.3f}"
    else:
        title_extra = ""
    plt.title(f"2D Embedding Visualization ({method_name}) for Numeric Tokens\n"
              f"Operation: {config.operation}, Prime: {config.prime}{title_extra}")
    plt.xlim(x_min - margin_x, x_max + margin_x)
    plt.ylim(y_min - margin_y, y_max + margin_y)
    plt.grid(True)

    # Save the plot.
    os.makedirs('results', exist_ok=True)
    op_label = config.operation.replace("/", "_mod_div_").replace("//", "_euc_div_")
    save_path = os.path.join('results', f"embedding_plot_{method_name}_{op_label}_{config.prime}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Embedding plot saved to {save_path}")


def save_embedding_snapshot(model, config, snapshot_idx, method="PCA", val_acc=None):
    """
    Extracts the learned embeddings from the model's embedding layer,
    reduces them to 2D using PCA or t-SNE (default: t-SNE), centers the projection,
    and saves a snapshot plot.
    
    Each token (0 to prime-1) is plotted as a text label colored by its normalized index.
    The projection is centered (mean subtracted) and the axes are fixed to be symmetric.
    The snapshot is saved with the snapshot index for later animation.
    
    If val_acc is provided, it is included in the title.
    """
    import numpy as np
    # --- Extract embeddings ---
    if hasattr(model, 'token_embeddings'):
        embeddings = model.token_embeddings.weight.detach().cpu()
    elif hasattr(model, 'embedding'):
        embeddings = model.embedding.weight.detach().cpu()
    else:
        raise ValueError("The model does not have an accessible embedding layer.")
    
    prime = config.prime
    numeric_embeddings = embeddings[:prime]  # shape: [prime, dim_model]
    
    # --- Dimensionality reduction ---
    if method.upper() == "PCA":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        proj = reducer.fit_transform(numeric_embeddings.numpy())
        method_name = "PCA"
    elif method.upper() in ["TSNE", "T-SNE"]:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        proj = reducer.fit_transform(numeric_embeddings.numpy())
        method_name = "t-SNE"
    else:
        raise ValueError("Invalid method. Choose 'PCA' or 't-SNE'.")
    
    # --- Center the projection ---
    center = np.mean(proj, axis=0)
    proj_centered = proj - center
    
    # --- Fix axis limits: set symmetric limits based on max absolute value ---
    max_val = np.max(np.abs(proj_centered))
    margin = max_val * 0.2
    lim = max_val + margin
    
    # --- Plot the snapshot ---
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("viridis")
    for i, (x, y) in enumerate(proj_centered):
        # Color tokens by increasing index normalized to [0,1]
        color = cmap(i / (prime - 1) if prime > 1 else 0)
        plt.text(x, y, str(i), fontsize=12, color=color, weight='bold')
    plt.xlabel(f"{method_name} Component 1")
    plt.ylabel(f"{method_name} Component 2")
    title_str = (f"{method_name} Projection of Embeddings\n"
                 f"Operation: {config.operation}, Prime: {prime}\n"
                 f"Gradient Step: {snapshot_idx}")
    if val_acc is not None:
        title_str += f", Val Acc: {val_acc:.3f}"
    plt.title(title_str)
    plt.xlim([-lim, lim])
    plt.ylim([-lim, lim])
    
    import os
    os.makedirs("snapshots", exist_ok=True)
    op_label = config.operation.replace("/", "_mod_div_").replace("//", "_euc_div_")
    filename = f"embedding_snapshot_{method_name}_{op_label}_{prime}_{snapshot_idx:04d}.png"
    save_path = os.path.join("snapshots", filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved snapshot {snapshot_idx} to {save_path}")