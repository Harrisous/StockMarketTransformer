'''Reference: https://github.com/deepseek-ai/DeepSeek-V3/tree/main/inference'''
import torch
import torch.nn as nn
import math

# --- Router and Expert Modules ---

class Router(nn.Module):
    def __init__(self, embed_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(embed_dim, num_experts)

    def forward(self, x):
        logits = self.gate(x)  # [batch, seq_len, num_experts]
        # print("router logits min/max:", logits.min().item(), logits.max().item())
        weights = torch.softmax(logits, dim=-1)
        # print("router weights min/max:", weights.min().item(), weights.max().item())
        return weights

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        # print("FeedForward after fc1:", x.min().item(), x.max().item())
        x = self.dropout(x)
        x = self.fc2(x)
        # print("FeedForward after fc2:", x.min().item(), x.max().item())
        return x

class Expert(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.ff = FeedForward(embed_dim, hidden_dim, dropout)

    def forward(self, x):
        return self.ff(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # print("Q min/max:", Q.min().item(), Q.max().item())
        # print("K min/max:", K.min().item(), K.max().item())
        # print("V min/max:", V.min().item(), V.max().item())
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        # scores = torch.clamp(scores, min=-30, max=30)
        # print("scores min/max:", scores.min().item(), scores.max().item())
        
        attn_weights = torch.softmax(scores, dim=-1)
        # print("attn_weights min/max:", attn_weights.min().item(), attn_weights.max().item())
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.out(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1, num_shared_experts=2, num_specialized_experts=6):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.shared_experts = nn.ModuleList([Expert(embed_dim, hidden_dim, dropout) for _ in range(num_shared_experts)])
        self.specialized_experts = nn.ModuleList([Expert(embed_dim, hidden_dim, dropout) for _ in range(num_specialized_experts)])
        self.num_experts = num_shared_experts + num_specialized_experts
        self.router = Router(embed_dim, self.num_experts)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        router_weights = self.router(x)  # [batch, seq_len, num_experts]
        expert_outputs = []
        for expert in self.shared_experts:
            expert_outputs.append(expert(x))
        for expert in self.specialized_experts:
            expert_outputs.append(expert(x))
        # [num_experts, batch, seq_len, embed_dim] -> [batch, seq_len, num_experts, embed_dim]
        expert_outputs = torch.stack(expert_outputs, dim=2)  # [batch, seq_len, num_experts, embed_dim]
        router_weights = router_weights.unsqueeze(-1)        # [batch, seq_len, num_experts, 1]
        ff_output = (expert_outputs * router_weights).sum(dim=2)  # [batch, seq_len, embed_dim]
        x = self.norm2(x + ff_output)
        return x

class TransformerForMultiStepPrediction(nn.Module):
    def __init__(
        self,
        input_dim=8,
        output_dim=4,
        embed_dim=32,
        num_heads=4,
        num_layers=4,
        hidden_dim=64,
        num_steps=7, # the step range to forecast
        dropout=0.1,
        num_shared_experts=2,
        num_specialized_experts=6,
        seq_len=252
    ):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, embed_dim)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
                num_shared_experts=num_shared_experts,
                num_specialized_experts=num_specialized_experts
            ) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, num_steps * output_dim)
        self.num_steps = num_steps
        self.output_dim = output_dim
        self.seq_len = seq_len

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, x):
        device = x.device
        batch_size, seq_len, _ = x.size()
        x = self.fc_in(x)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = x + self.position_embedding(positions)
        mask = self.generate_square_subsequent_mask(seq_len).to(device)
        for block in self.blocks:
            x = block(x, mask=None) # mask is being disabled
        x_last = x[:, -1, :]  # Take only the last time step
        output = self.fc_out(x_last)  # [batch_size, num_steps * output_dim]
        output = output.view(batch_size, self.num_steps, self.output_dim)  # [batch, num_steps, output_dim]
        return output

