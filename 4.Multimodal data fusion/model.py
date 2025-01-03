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
        a = a.permute(1, 0, 2)    # (seq_len_y, batch_size, dim)
        y = y.permute(1, 0, 2)  # (seq_len_y, batch_size, dim)
        attn_output, attn_weights = self.cross_attn(y, a)

        y2 = y + self.dropout3(attn_output)
        y2 = self.norm3(y2)  # Apply LayerNorm to the output

        ff_output = self.linear2(self.dropout(self.linear1(y2)))  # (seq_len_y, batch_size, dim)
        y2 = y2 + self.dropout4(ff_output)
        y2 = self.norm4(y2)  # Final normalization

        y2 = y2.permute(1, 0, 2)  # (batch_size, seq_len_y, dim)

        return y2, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers,dim, num_heads, dim_feedforward, dropout,num_classes):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderlayer(dim, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(512, 128)
        self.fc_out = nn.Linear(dim, num_classes)

    def forward(self, go, seq,img):
        go=self.fc1(go)
        img = self.fc2(img)
        memory=torch.cat((seq,img),dim=1)
        for layer in self.layers:
            src, attn_weights = layer(go, memory)
        src=torch.squeeze(src)
        x = self.fc_out(src)
        return x, attn_weights
