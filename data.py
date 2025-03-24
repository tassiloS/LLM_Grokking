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
    If the operation string contains 'z', the returned function will expect three components.
    Otherwise, it will expect two components.
    
    In our implementation:
      - The "/" operator is interpreted as modular division via mod_div.
      - The "//" operator is left as usual integer division.
    """
    if 'z' in operation:
        def op_func(x, y, z, p):
            op_str = re.sub(r'(?<!/)(\S+)\s*/\s*(\S+)(?!/)', r'mod_div(\1, \2, p)', operation)
            env = {
                'x': x,
                'y': y,
                'z': z,
                'p': p,
                'torch': torch,
                'min': torch.minimum,
                'max': torch.maximum,
                'mod_div': mod_div
            }
            result = eval(op_str, {"__builtins__": {}}, env)
            result = result % p
            return x, y, z, result
        return op_func
    else:
        def op_func(x, y, p):
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
            result = result % p
            return x, y, result
        return op_func

def operation_mod_p_data(operation: str, p: int, eq_token: int, op_token: int):
    """
    Generates data for an operation modulo p.
    
    - For two-component operations (using only x and y), it creates a Cartesian product
      of values for x and y (with special handling for "x/y" to avoid division by zero).
      Inputs are packed as [x, op_token, y, eq_token].
      
    - For three-component operations (if the operation string contains "z"),
      it creates a Cartesian product for x, y, and z, and packs the inputs as
      [x, op_token, y, op_token, z, eq_token].
    """
    if 'z' in operation:
        # Three-component case: x, y, and z all range from 0 to p-1.
        x = torch.arange(0, p)
        y = torch.arange(0, p)
        z = torch.arange(0, p)
        x, y, z = torch.cartesian_prod(x, y, z).T
        op_func = generate_operation_func(operation)
        x, y, z, labels = op_func(x, y, z, p)
        labels = labels.to(torch.long)
        op = torch.ones_like(x) * op_token
        eq = torch.ones_like(x) * eq_token
        # Pack as: [x, op_token, y, op_token, z, eq_token]
        inputs = torch.stack([x, op, y, op, z, eq], dim=1)
    else:
        # Two-component case as before.
        x = torch.arange(0, p)
        if operation.strip() == "x/y":
            y = torch.arange(1, p)
        else:
            y = torch.arange(0, p)
        x, y = torch.cartesian_prod(x, y).T
        op_func = generate_operation_func(operation)
        x, y, labels = op_func(x, y, p)
        labels = labels.to(torch.long)
        op = torch.ones_like(x) * op_token
        eq = torch.ones_like(x) * eq_token
        inputs = torch.stack([x, op, y, eq], dim=1)
    return inputs, labels

def get_data(operation: str, prime: int, training_fraction: float, batch_size: int):
    """
    Generates training and validation data loaders for a custom operation modulo a prime.
    
    The operation is specified as a string (e.g. "x+min(x,y)" or "x+y+z") and is evaluated
    to compute the label for each input tuple.
    
    The eq_token and op_token are set to prime and prime+1 respectively.
    This function works with either two- or three-component tasks.
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
