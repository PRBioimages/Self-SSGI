import torch
import math
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=2699):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] =torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)

def get_mask(x,mask,ratio):
    masked_pos_all=torch.zeros(x.shape[0],x.shape[1])
    mask_x = x.clone()
    for i in range(len(mask)):
        current_mask=mask[i]
        false_positions = torch.nonzero(current_mask == False)
        max_pos=torch.max(false_positions)
        mask_percentage = ratio
        num_masked_positions = int(mask_percentage * max_pos)
        masked_pos = random.sample(range(max_pos), num_masked_positions)
        masked_pos_all[i, masked_pos] = 1
        for j in range(len(masked_pos)):
            mask_x[i, masked_pos[j]] =0
    label= x[masked_pos_all == 1]



    return mask_x,masked_pos_all,label


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(0.1)#0.1
        # Linear layers for query, key, and value projections
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # Final linear layer for concatenated outputs from all heads
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask_ini=None):
        batch_size = query.shape[0]

        # Project inputs into query, key, and value spaces
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # Split the heads
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Initialize an empty list to store attention scores for each head
        attention_scores = []
        mask=mask_ini.clone()##这个必须有，否则会报错
        #mask = mask.unsqueeze(1).expand(-1, query.shape[1], -1)
        mask = mask.unsqueeze(1)

        attention_scores = []
        weighted_sums = []
        # Apply mask to each head
        for i in range(self.n_heads):
            # Extract the i-th head
            Q_i, K_i, V_i = Q[:, i, :, :], K[:, i, :, :], V[:, i, :, :]

            # Calculate attention scores for the i-th head
            energy = torch.matmul(Q_i, K_i.permute(0, 2, 1)) / self.head_dim
            #aa=energy.detach().cpu().numpy()
            #cc=mask.detach().cpu().numpy()
            # Apply mask (optional)
            #cc=~mask_ini.bool().unsqueeze(1)
            if mask is not None:
                energy = energy *mask
                #bb=energy.detach().cpu().numpy()
                energy = energy.masked_fill(
                    ~mask_ini.bool().unsqueeze(1),
                    float("-inf"),
                )


            # Apply softmax to obtain attention weights for the i-th head
            attention_i = F.softmax(energy, dim=-1)
            #attention_i[torch.isnan(attention_i)] = 0

            # Apply dropout (optional)
            attention_i = self.dropout(attention_i)

            # Obtain the weighted sum of values for the i-th head
            # Obtain the weighted sum of values for the i-th head
            #aa=attention_i.detach().cpu().numpy()
            #bb=V_i.detach().cpu().numpy()
            weighted_sum_i = torch.matmul(attention_i, V_i)

            # Append attention scores and weighted sum for the i-th head
            attention_scores.append(attention_i)
            weighted_sums.append(weighted_sum_i)

        # Concatenate attention scores from all heads
        attention_scores = torch.stack(attention_scores, dim=1)


        # Concatenate weighted sums from all heads
        weighted_sums = torch.stack(weighted_sums, dim=1)


        # Linear layer to obtain the final output
        output = self.fc_out(weighted_sums.view(batch_size, -1, self.d_model))

        return output



class Feedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(Feedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention =MultiHeadAttention(d_model, num_heads)
        self.feedforward = Feedforward(d_model, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_key_padding_mask=None):
        # Self Attention
        attn_output = self.self_attention(x,x,x,src_key_padding_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)


        # Feedforward
        ff_output = self.feedforward(x)
        x = x+self.dropout2(ff_output)
        x = self.norm2(x)


        return x




class Seq_Struce_Bert(nn.Module):
    def __init__(self, embed_size, sequence_length, heads, num_layers, num_classes, device, dropout):
        super(Seq_Struce_Bert, self).__init__()
        self.device = device
        self.sequence_length = sequence_length
        self.positional_encoder = PositionalEncoding(embed_size, dropout, sequence_length).to(self.device)
        self.embed_tokens = nn.Embedding(20, embed_size-15)
        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=embed_size, num_heads=heads,d_ff=2048
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size,num_classes )


    def forward(self, x, mask,ratio,arg1,arg2):
        mask_x, masked_pos_all, label = get_mask(x, mask,ratio)
        amino = mask_x[:, :, -1]
        amino = amino.to(torch.int)
        embed_amino_ini = self.embed_tokens(amino.long())
        min_vals, _ = embed_amino_ini.min(dim=-1, keepdim=True)
        max_vals, _ = embed_amino_ini.max(dim=-1, keepdim=True)


        embed_amino= (embed_amino_ini - min_vals) / (max_vals - min_vals)


        before_fea = mask_x[:, :, :-1]
        final_fea = torch.cat((before_fea, embed_amino), dim=2)
        final_fea = self.positional_encoder(final_fea)
        final_fea = final_fea.to(torch.float32)
        label = label.to(torch.float32)
        mask_1=~mask
        mask_1=mask_1.float()
        for i, layer in enumerate(self.transformer_layers):
            if i == 0:
                mask_1[masked_pos_all.nonzero(as_tuple=True)] = arg1

            elif i == 1:
                mask_1[masked_pos_all.nonzero(as_tuple=True)] = arg2

            else:
                mask_1[masked_pos_all.nonzero(as_tuple=True)] = 1

            final_fea = layer(final_fea, src_key_padding_mask=mask_1)


        output_ini = final_fea[masked_pos_all == 1]
        output = self.fc_out(output_ini)
        L1_output = output[:, :15]
        CE_output = output[:, 15:]

        return L1_output, CE_output, label, final_fea, mask








