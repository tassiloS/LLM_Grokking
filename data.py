from math import ceil
import torch
import re

def mod_pow_tensor(y, exp, mod):
    """
    Compute y^exp (mod mod) elementwise using exponentiation by squaring.
    Assumes y is a torch tensor of integers.
    """
    result = torch.ones_like(y)
    base = y % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp //= 2
    return result

def mod_div(x, y, p):
    """
    Compute x/y (mod p) as x * (y^(p-2)) mod p.
    Uses Fermat's Little Theorem (p is prime) to compute the inverse of y.
    """
    inv_y = mod_pow_tensor(y, p-2, p)
    return (x * inv_y) % p

def generate_operation_func(operation: str):
    """
    Returns a function that computes a custom operation.
    The returned function takes tensors x, y, and an integer p, and returns (x, y, result),
    where result is computed by evaluating the given operation string in a restricted environment.
    After evaluation, the result is reduced modulo p so that it is in the range [0, p).

    In our implementation:
      - The "/" operator in the operation string is interpreted as modular division (x/y mod p).
      - The "//" operator is left as the usual integer (Euclidean) division.
    """
    def op_func(x, y, p):
        # Preprocess the operation string:
        # Replace every occurrence of a single "/" (not part of a "//") with a call to mod_div.
        # The regex matches two tokens separated by a "/" and replaces with: mod_div(token1, token2, p)
        op_str = re.sub(r'(?<!/)(\S+)\s*/\s*(\S+)(?!/)', r'mod_div(\1, \2, p)', operation)
        env = {
            'x': x,
            'y': y,
            'p': p,
            'torch': torch,
            'min': torch.minimum,
            'max': torch.maximum,
            'mod_div': mod_div
        }
        result = eval(op_str, {"__builtins__": {}}, env)
        # Reduce the result modulo p so that the result is in the range [0, p)
        result = result % p
        return x, y, result
    return op_func

def operation_mod_p_data(operation: str, p: int, eq_token: int, op_token: int):
    """
    Generates data for the operation x ◦ y (mod p).

    For x: values from 0 to p-1.
    For y: values from 0 to p-1, unless the operation is exactly "x/y" (to avoid division by zero),
    in which case y is in [1, p).

    The input examples are built by stacking:
        [x, op_token, y, eq_token]
    and the labels are computed using the custom operation provided by the user.
    """
    x = torch.arange(0, p)
    if operation.strip() == "x/y":
        y = torch.arange(1, p)
    else:
        y = torch.arange(0, p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token

    op_func = generate_operation_func(operation)
    x, y, labels = op_func(x, y, p)
    
    # Ensure labels are of type Long for use with CrossEntropyLoss
    labels = labels.to(torch.long)

    inputs = torch.stack([x, op, y, eq], dim=1)

    return inputs, labels

def get_data(operation: str, prime: int, training_fraction: float, batch_size: int):
    """
    Generates training and validation data loaders for a custom operation modulo a prime.
    
    The operation is specified as a string (e.g. "x+min(x,y)") and is evaluated
    to compute the label for each (x, y) pair.
    
    The eq_token and op_token are set to prime and prime+1 respectively.
    """
    inputs, labels = operation_mod_p_data(operation, prime, prime, prime+1)
    dataset = torch.utils.data.TensorDataset(inputs, labels)

    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = min(batch_size, ceil(len(dataset) / 2))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader
