"""Small transformer for modular arithmetic experiments."""
import torch
import torch.nn as nn
import math


class ModularArithmeticTransformer(nn.Module):
    """1-layer transformer for learning modular arithmetic (a op b = c mod p)."""

    def __init__(self, p=97, d_model=128, n_heads=4, d_ff=512, dropout=0.0):
        super().__init__()
        self.p = p
        self.d_model = d_model

        # Embedding: tokens are integers 0..p-1 for operands, p for operator, p+1 for equals
        self.token_embed = nn.Embedding(p + 2, d_model)
        self.pos_embed = nn.Embedding(5, d_model)  # positions: a, op, b, =, result

        # Transformer layer
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Output head: predict result token (0..p-1)
        self.output_head = nn.Linear(d_model, p)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """x: (batch, 4) containing [a, op_token, b, eq_token]"""
        B, S = x.shape
        positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)

        h = self.token_embed(x) + self.pos_embed(positions)
        h = self.dropout(h)

        # Self-attention
        attn_out, _ = self.attn(h, h, h)
        h = self.ln1(h + attn_out)

        # Feed-forward
        ff_out = self.ff(h)
        h = self.ln2(h + ff_out)

        # Predict from last position
        logits = self.output_head(h[:, -1, :])
        return logits


def create_modular_dataset(p=97, operation='add', frac_train=0.5, seed=42):
    """Create modular arithmetic dataset: (a op b) mod p.

    Returns: train_data, test_data as lists of (a, b, result) tuples.
    """
    torch.manual_seed(seed)

    all_pairs = [(a, b) for a in range(p) for b in range(p)]

    # Shuffle deterministically
    import random
    rng = random.Random(seed)
    rng.shuffle(all_pairs)

    n_train = int(len(all_pairs) * frac_train)
    train_pairs = all_pairs[:n_train]
    test_pairs = all_pairs[n_train:]

    def compute(a, b):
        if operation == 'add':
            return (a + b) % p
        elif operation == 'sub':
            return (a - b) % p
        elif operation == 'mul':
            return (a * b) % p
        elif operation == 'div':
            if b == 0:
                return None
            return (a * pow(b, p - 2, p)) % p
        raise ValueError(f"Unknown operation: {operation}")

    op_token = p  # special token for operator
    eq_token = p + 1  # special token for equals

    def make_tensors(pairs):
        inputs, labels = [], []
        for a, b in pairs:
            result = compute(a, b)
            if result is None:
                continue
            inputs.append([a, op_token, b, eq_token])
            labels.append(result)
        return torch.tensor(inputs), torch.tensor(labels)

    train_x, train_y = make_tensors(train_pairs)
    test_x, test_y = make_tensors(test_pairs)

    return train_x, train_y, test_x, test_y
