from math import ceil
import torch
import re
import itertools
import random

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
    (This only works correctly when p is prime.)
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

#######################
# Helpers for random abelian groups

def factorize(n):
    """Return a list of (prime, exponent) for n using trial division."""
    factors = []
    d = 2
    while d * d <= n:
        count = 0
        while n % d == 0:
            count += 1
            n //= d
        if count > 0:
            factors.append((d, count))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return factors

def partitions(n):
    """
    Returns all partitions of the positive integer n.
    Each partition is returned as a tuple of positive integers in nonincreasing order.
    """
    if n == 0:
        return [()]
    result = []
    def helper(n, max_value, current):
        if n == 0:
            result.append(tuple(current))
        else:
            for i in range(min(max_value, n), 0, -1):
                helper(n - i, i, current + [i])
    helper(n, n, [])
    return result

def direct_product_group(moduli):
    """
    Given a list of moduli, returns a list of tuples representing the direct product
    of cyclic groups Z_{m} for each m in moduli.
    """
    groups = [list(range(m)) for m in moduli]
    return [tuple(x) for x in itertools.product(*groups)]

def group_add(elem1, elem2, moduli):
    """
    Coordinate-wise addition of two group elements (tuples) modulo the provided moduli.
    """
    return tuple((a + b) % mod for a, b, mod in zip(elem1, elem2, moduli))

def random_abelian_group(p):
    """
    Selects a random abelian group of order p among all isomorphism classes.
    
    Returns a tuple (f, f_inv, moduli) where:
      - f is a list of group elements (each represented as a tuple) of length p,
        representing the elements of the group, with f[0] equal to the identity.
      - f_inv is a dictionary mapping each group element (tuple) to its index.
      - moduli is a list of moduli corresponding to each coordinate in the group's representation.
    """
    # Factorize p into prime powers.
    factors = factorize(p)  # list of (q, exponent)
    overall_moduli = []  # will hold moduli for each cyclic factor across all primes.
    group_components = []  # will hold the list of group elements for each prime factor.
    for (q, exp) in factors:
        parts = partitions(exp)  # partitions of the exponent
        chosen_partition = random.choice(parts)  # randomly select one partition
        # For each part r, the corresponding cyclic factor is Z_{q^r}
        moduli_component = [q**r for r in chosen_partition]
        overall_moduli.extend(moduli_component)
        group_comp = direct_product_group(moduli_component)
        group_components.append(group_comp)
    # The overall group is the direct product over the prime factors.
    overall_elements = []
    for element in itertools.product(*group_components):
        # Each element is a tuple of tuples; flatten it.
        flattened = tuple(x for comp in element for x in comp)
        overall_elements.append(flattened)
    assert len(overall_elements) == p, f"Group order mismatch: got {len(overall_elements)} elements, expected {p}."
    identity = tuple(0 for _ in overall_moduli)
    # Create a random bijection f from indices 0,...,p-1 to overall_elements, with f[0] = identity.
    elements_copy = overall_elements.copy()
    elements_copy.remove(identity)
    random.shuffle(elements_copy)
    f = [identity] + elements_copy
    f_inv = {elem: i for i, elem in enumerate(f)}
    return f, f_inv, overall_moduli

#######################
# Dataset Generation

def operation_mod_p_data(operation: str, p: int, eq_token: int, op_token: int):
    """
    Generates data for an operation modulo p.
    
    Special case: if the operation (after stripping) is "abelian",
    then generate a random abelian group of order p selected uniformly among
    all isomorphism classes.
    
    For the "abelian" operation:
      - A random abelian group of order p is chosen.
      - The group operation is defined as:
            x ∘ y = f( f⁻¹(x) + f⁻¹(y) )
        where f is the random bijection produced by random_abelian_group.
      - Inputs are packed as [x, op_token, y, eq_token].
      
    For non-abelian tasks:
      - For two-component operations (using x and y), inputs are built as:
            [x, op_token, y, eq_token]
      - For three-component operations (if "z" appears in the operation string),
            the inputs are built as:
            [x, op_token, y, op_token, z, eq_token].
    """
    if operation.strip() == "abelian":
        f, f_inv, moduli = random_abelian_group(p)
        x = torch.arange(0, p)
        y = torch.arange(0, p)
        x, y = torch.cartesian_prod(x, y).T
        labels = []
        for i in range(x.shape[0]):
            a = int(x[i].item())
            b = int(y[i].item())
            elem_a = f[a]
            elem_b = f[b]
            sum_elem = group_add(elem_a, elem_b, moduli)
            labels.append(f_inv[sum_elem])
        labels = torch.tensor(labels, dtype=torch.long)
        op_tensor = torch.ones_like(x) * op_token
        eq_tensor = torch.ones_like(x) * eq_token
        inputs = torch.stack([x, op_tensor, y, eq_tensor], dim=1)
        return inputs, labels

    if 'z' in operation:
        x = torch.arange(0, p)
        y = torch.arange(0, p)
        z = torch.arange(0, p)
        x, y, z = torch.cartesian_prod(x, y, z).T
        op_func = generate_operation_func(operation)
        x, y, z, labels = op_func(x, y, z, p)
        labels = labels.to(torch.long)
        op = torch.ones_like(x) * op_token
        eq = torch.ones_like(x) * eq_token
        inputs = torch.stack([x, op, y, op, z, eq], dim=1)
        return inputs, labels
    else:
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
    Generates training and validation data loaders for a custom operation modulo a number.
    
    The operation is specified as a string (e.g. "x+min(x,y)", "x+y+z", or "abelian")
    and is evaluated to compute the label for each input tuple.
    
    In the special case "abelian", a random abelian group of order prime is generated
    (with a truly random isomorphism class chosen among all possibilities).
    
    For non-abelian tasks, the tokens are set up as:
        [x, op_token, y, eq_token]  (or with an extra z if needed)
    
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

def check_abelian(get_data_func, prime: int, training_fraction: float, batch_size: int):
    """
    Checks if the dataset generated by get_data (when the operation is "abelian")
    satisfies the abelian (i.e., commutative) property along with the other group axioms:
      - Commutativity: For every pair (x, y), label(x,y) == label(y,x).
      - Associativity: For all a, b, c, label(label(a, b), c) == label(a, label(b, c)).
      - Identity: There exists an element e such that for every a, label(e, a) == a and label(a, e) == a.
      - Inverses: For every a, there exists b such that label(a, b) == e and label(b, a) == e.
    
    Returns True if the operation forms an abelian group, False otherwise.
    """
    # Get the data loaders. We'll use the training loader.
    train_loader, _ = get_data_func("abelian", prime, training_fraction, batch_size)
    
    # Get the underlying TensorDataset.
    dataset = train_loader.dataset
    if hasattr(dataset, 'dataset'):
        dataset = dataset.dataset  # In case it's a Subset.
    
    # Extract inputs and labels.
    inputs, labels = dataset.tensors  # inputs shape: [N, 4], labels shape: [N]
    
    # Build a dictionary mapping (x, y) -> label.
    mapping = {}
    for i in range(inputs.shape[0]):
        x_val = int(inputs[i, 0].item())
        y_val = int(inputs[i, 2].item())
        mapping[(x_val, y_val)] = int(labels[i].item())
    
    # 1. Check commutativity: for each (x, y), verify label(x,y) == label(y,x)
    for (x_val, y_val), label in mapping.items():
        if (y_val, x_val) not in mapping or mapping[(y_val, x_val)] != label:
            return False

    # Determine the set of group elements.
    group_elements = set()
    for (x, y) in mapping.keys():
        group_elements.add(x)
        group_elements.add(y)
    for val in mapping.values():
        group_elements.add(val)
    
    # 2. Check associativity: for all a, b, c, verify:
    #    mapping[(mapping[(a, b)], c)] == mapping[(a, mapping[(b, c)])]
    for a in group_elements:
        for b in group_elements:
            for c in group_elements:
                if (a, b) not in mapping or (b, c) not in mapping:
                    return False
                first = mapping[(a, b)]
                if (first, c) not in mapping:
                    return False
                second = mapping[(b, c)]
                if (a, second) not in mapping:
                    return False
                if mapping[(first, c)] != mapping[(a, second)]:
                    return False

    # 3. Find the identity element: e such that for every a in group_elements,
    #    mapping[(e, a)] == a and mapping[(a, e)] == a.
    identity = None
    for e in group_elements:
        is_identity = True
        for a in group_elements:
            if (e, a) not in mapping or (a, e) not in mapping:
                is_identity = False
                break
            if mapping[(e, a)] != a or mapping[(a, e)] != a:
                is_identity = False
                break
        if is_identity:
            identity = e
            break
    if identity is None:
        return False

    # 4. Check existence of inverses: for every a in group_elements,
    #    there exists b in group_elements such that mapping[(a, b)] == identity and mapping[(b, a)] == identity.
    for a in group_elements:
        has_inverse = False
        for b in group_elements:
            if (a, b) in mapping and (b, a) in mapping:
                if mapping[(a, b)] == identity and mapping[(b, a)] == identity:
                    has_inverse = True
                    break
        if not has_inverse:
            return False

    return True

