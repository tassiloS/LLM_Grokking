from argparse import ArgumentParser
from training import main
import torch

if __name__ == "__main__":

    default_device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

    parser = ArgumentParser()
    parser.add_argument("--operation", type=str, default="x/y")
    parser.add_argument("--training_fraction", type=float, default=0.5)
    parser.add_argument("--prime", type=int, default=97)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1)
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--device", type=str, default=default_device)
    args = parser.parse_args()

    main(args)
