import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)

    def forward(self, X, Y):
        Q = self.query(X)  # (batch_size, seq_len_x, dim)
        K = self.key(Y)    # (batch_size, seq_len_y, dim)
        V = self.value(Y)  # (batch_size, seq_len_y, dim)

        attn_output, attn_weights = self.attention(Q, K, V)
        return attn_output, attn_weights

class TransformerEncoderlayer(nn.Module):
    def __init__(self, dim, num_heads, dim_feedforward, dropout):
        super(TransformerEncoderlayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.cross_attn = CrossAttention(dim, num_heads)
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)


    def forward(self, y, a):
        # a: (batch_size, seq_len_a, dim)
        # b: (batch_size, seq_len_b, dim)
        # y: (batch_size, seq_len_y, dim)

        a = a.permute(1, 0, 2)    # (seq_len_y, batch_size, dim)
        y = y.permute(1, 0, 2)  # (seq_len_y, batch_size, dim)

        # 1. Self-attention on y (y attends to itself)
        #y_self_attn_output, _ = self.self_attn(y, y, y)  # (seq_len_y, batch_size, dim)
        #y = y + self.dropout1(y_self_attn_output)  # Residual connection for self-attention
        #y = self.norm1(y)  # Layer normalization after self-attention

        #a_self_attn_output, _ = self.self_attn(a, a, a)  # (seq_len_y, batch_size, dim)
        #a = a + self.dropout2(a_self_attn_output)  # Residual connection for self-attention
        #a = self.norm2(a)  # Layer normalization after self-attention


        # Cross-attention from a and b to y
        attn_output, attn_weights = self.cross_attn(y, a)






        # Residual connection and layer normalization
        y2 = y + self.dropout3(attn_output)
        y2 = self.norm3(y2)  # Apply LayerNorm to the output

        # Feed-forward network
        ff_output = self.linear2(self.dropout(self.linear1(y2)))  # (seq_len_y, batch_size, dim)
        y2 = y2 + self.dropout4(ff_output)
        y2 = self.norm4(y2)  # Final normalization

        # Permute back to the original shape
        y2 = y2.permute(1, 0, 2)  # (batch_size, seq_len_y, dim)

        return y2, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers,dim, num_heads, dim_feedforward, dropout,num_classes):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderlayer(dim, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(dim, num_classes)

    def forward(self, src,memory):
        for layer in self.layers:
            src, attn_weights = layer(src, memory)


        src=torch.squeeze(src)
        x = self.fc_out(src)
        return x, attn_weights
