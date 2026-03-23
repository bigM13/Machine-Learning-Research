#from s5 import S5
import torch
from torch import nn

class ChaosUnit(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.w1 = nn.Parameter(torch.randn(hidden_size, hidden_size) * 1.25)
        self.w2 = nn.Parameter(torch.randn(hidden_size, hidden_size) * 4.0)
        self.b1 = nn.Parameter(torch.zeros(hidden_size))
        self.b2 = nn.Parameter(torch.zeros(hidden_size))
        self.w_input = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)

        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.w_input)

    def forward(self, x_input, prev_state):
        part1 = torch.relu(x_input @ self.w_input.T + prev_state @ self.w1.T + self.b1)
        part2 = torch.relu(prev_state @ self.w2.T - self.b2)
        return torch.relu(part1 - part2)
    
class ChaosUnitClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_classes, max_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.chaos_unit = ChaosUnit(hidden_size=hidden_size, input_size=hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_classes)
        self.max_length = max_length
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # input_ids: [batch_size, seq_len]
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, hidden_size]

        batch_size, seq_len, _ = embedded.size()
        device = embedded.device

        # Initial hidden state
        h_t = torch.zeros(batch_size, self.hidden_size, device=device)

        for t in range(seq_len):
            x_t = embedded[:, t, :]
            h_t = self.chaos_unit(x_t, h_t)

        logits = self.output_layer(h_t)  # [batch_size, num_classes]
        return logits


#######
# This is new archetype for making models

class S5InspiredRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.Wxh = nn.Linear(input_size, hidden_size, bias=True)
        self.Whh = nn.Linear(hidden_size, hidden_size, bias=True)
        self.tanh = nn.Tanh()

        # 🔹 Single learnable decay parameter (S5-style continuous update)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, h0=None):
        if not self.batch_first:
            x = x.transpose(0, 1)  # (seq, batch, input) → (batch, seq, input)

        batch_size, seq_len, _ = x.size()
        alpha = torch.sigmoid(self.alpha)

        # Initialize hidden state if not provided
        if h0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h = h0

        outputs = []  # collect all time step outputs

        for t in range(seq_len):
            h_tilde = self.tanh(self.Wxh(x[:, t]) + self.Whh(h))
            h = (1 - alpha) * h + alpha * h_tilde
            outputs.append(h.unsqueeze(1))  # (B, 1, H)

        output = torch.cat(outputs, dim=1)  # (B, T, H)
        h_n = h.unsqueeze(0)  # (1, B, H) like nn.RNN

        return output, h_n



class RNNMockingTransformer(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, num_classes=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim) # instead of embedding
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        #self.rnn = S5InspiredRNN(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        #self.sigmoid = nn.Sigmoid()  # <- explicit sigmoid

    def generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


    def forward(self, input_ids, attention_mask=None, hidden=None):
        x = input_ids # needed because no embedding layer
        if x.dim() == 4:  # (B, T, K, D)
            x = x.sum(dim=2)  # -> (B, T, D)

        # Handle edge case (B, D) → (B, 1, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.input_proj(x.float())

        _, T, _ = x.size() # B, T, D
        attn_mask = self.generate_causal_mask(T, x.device)
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = x + attn_output  # residual connection
        
        output, hidden = self.rnn(x, hidden)
        logits = self.fc(output[:, -1, :])
        #logits = self.fc(x[:, -1, :]) # This line was to run experiment without RNN

        return logits
    

"""
Known working version that I do not want to risk editing and losing
class RNNMockingTransformer(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, num_classes=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim) # instead of embedding
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        #self.sigmoid = nn.Sigmoid()  # <- explicit sigmoid

    def generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


    def forward(self, input_ids, attention_mask=None, hidden=None):
        x = input_ids # needed because no embedding layer
        if x.dim() == 4:  # (B, T, K, D)
            x = x.sum(dim=2)  # -> (B, T, D)

        # Handle edge case (B, D) → (B, 1, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.input_proj(x.float())

        _, T, _ = x.size() # B, T, D
        attn_mask = self.generate_causal_mask(T, x.device)
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = x + attn_output  # residual connection
        
        output, hidden = self.rnn(x, hidden)
        logits = self.fc(output[:, -1, :])

        return logits
"""


from mamba_ssm import Mamba  # or Mamba2
import math
"""
class MyMambaModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_classes, mamba_state_dim, mamba_conv_width=4, mamba_expand=2):
        super().__init__()
        # optional embedding / input projection
        self.input_proj = nn.Linear(input_dim, model_dim) if input_dim != model_dim else nn.Identity()
        # Mamba backbone
        self.mamba = Mamba(
            d_model=model_dim,
            d_state=mamba_state_dim,
            d_conv=mamba_conv_width,
            expand=mamba_expand,
        )
        # classifier head
        self.classifier = nn.Linear(model_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask=None, return_hidden=False):
        #x: tensor [batch, seq_len, input_dim] (or whatever your input is)
        #return_hidden: if True, return intermediate hidden state (if Mamba supports it)
        
        x = input_ids.float()

        # If doc_ret (K=2 per timestep), combine along feature axis
        if x.dim() == 4:  # (B, T, K, D)
            x = x.sum(dim=2)  # -> (B, T, D)

        # Handle edge case (B, D) → (B, D, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        h = self.input_proj(x)
        # Mamba block
        out = self.mamba(h)  # shape [batch, seq_len, model_dim]

        # maybe take last time-step, or average, whatever your task needs
        # Suppose classification on last step:
        last = out[:, -1, :]  # shape [batch, model_dim]
        logits = self.classifier(last)  # shape [batch, num_classes]
        probs = self.sigmoid(logits)  # professor wants explicit sigmoid
        if return_hidden:
            return probs, out
        return probs
"""

class MyMambaModel(nn.Module):
    def __init__(self,
                 input_dim=4,          # now patch_dim = 4
                 model_dim=128,        # a good default for laptop-medium
                 num_classes=2,
                 mamba_state_dim=32,
                 mamba_conv_width=4,
                 mamba_expand=2,
                 max_seq_len=256):     # now num patches
        super().__init__()

        # small input normalization
        self.input_norm = nn.LayerNorm(input_dim)

        # patch embedding: 4 -> model_dim
        self.embed = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

        # small dropout for stability
        self.dropout = nn.Dropout(0.1)

        # positional encoding for 256 tokens
        self.pos = nn.Parameter(torch.randn(1, max_seq_len, model_dim) * 0.01)

        # Mamba backbone
        self.pre_norm = nn.LayerNorm(model_dim)
        self.mamba = Mamba(
            d_model=model_dim,
            d_state=mamba_state_dim,
            d_conv=mamba_conv_width,
            expand=mamba_expand,
        )
        self.post_norm = nn.LayerNorm(model_dim)

        # pooling head (mean pool over time)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, input_ids, attention_mask=None, return_hidden=False):
        # input_ids: (B, T, patch_dim)
        x = input_ids.float()

        # normalize patch vector
        x = self.input_norm(x)

        # embed patches -> (B, T, model_dim)
        h = self.embed(x)
        h = self.dropout(h)

        # add pos enc
        h = h + self.pos[:, :h.size(1)]

        h = self.pre_norm(h)
        out = self.mamba(h)
        out = self.post_norm(out)

        # aggregate across time — mean pooling (respect attention_mask if needed)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  # (B, T, 1)
            out = out * mask
            denom = mask.sum(dim=1).clamp(min=1e-6)
            pooled = out.sum(dim=1) / denom
        else:
            pooled = out.mean(dim=1)

        logits = self.classifier(pooled)  # (B, num_classes)

        if return_hidden:
            return logits, out
        return logits




class SimpleTransformerBaseline(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, num_classes=2, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = input_ids
        if x.dim() == 4:
            x = x.sum(dim=2)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_proj(x.float())

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask[:, :x.size(1)] == 0

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        logits = self.fc(x[:, -1, :])
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim=1024,       # feature dimension per timestep
        num_classes=2,        # LRA classification tasks usually 2 classes
        d_model=128,          # hidden dimension
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_len=1024,
        pool="mean"           # "mean" or "last"
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
        self.pool = pool

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: (B, T, D)
        attention_mask: (B, T), 1 = valid token, 0 = padding
        """
        x = input_ids.float()  # ensure float32

        B, T, D = x.shape
        assert T == 1024, f"Unexpected sequence length: got {T}, expected 1024"
        assert D == 1, f"Expected feature dim = 1 (grayscale pixel), got {D}"

        # Project input features to model dimension
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        # Pass through transformer
        x = self.transformer(x, src_key_padding_mask=None)

        # Pooling
        if self.pool == "mean":
            x = x.mean(dim=1)
        else:  # last valid timestep
            x = x[:, -1]

        logits = self.fc(x)
        return logits



class LinearRecurrentLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_ih = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, hidden=None):
        # x: (B, T, D)
        B, T, _ = x.shape
        if hidden is None:
            hidden = x.new_zeros(B, self.W_hh.out_features)
        outputs = []
        for t in range(T):
            hidden = self.W_ih(x[:, t, :]) + self.W_hh(hidden)
            outputs.append(hidden.unsqueeze(1))
        return torch.cat(outputs, dim=1), hidden


class ProjToMultiLayerRNN(nn.Module):
    def __init__(self, vocab_size, input_dim, proj_dim, hidden_dim, num_layers=1, num_classes=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, proj_dim)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:  # even layers: RNN
                self.layers.append(nn.RNN(
                    input_size=proj_dim if i == 0 else hidden_dim,
                    hidden_size=hidden_dim,
                    batch_first=True
                ))
            else:  # odd layers: linear recurrent
                self.layers.append(LinearRecurrentLayer(hidden_dim, hidden_dim))

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask=None, hidden=None):
        x = input_ids
        x = x.float()
        # Always reduce (B, T, K, D) -> (B, T, D)
        if x.dim() == 4:
            x = x.sum(dim=2)

        # Convert (B, D) -> (B, 1, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_proj(x.float())

        # Process through the stacked layers
        for layer in self.layers:
            # ensure x is always 3D before passing to RNN
            if x.dim() == 4:
                x = x.squeeze(2)
            x, _ = layer(x)

        logits = self.fc(x[:, -1, :])
        return logits

"""
class MurmurHashAsRNN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Learnable constants
        self.c1 = nn.Parameter(torch.randn(hidden_dim))
        self.c2 = nn.Parameter(torch.randn(hidden_dim))
        self.c3 = nn.Parameter(torch.randn(hidden_dim))
        self.c4 = nn.Parameter(torch.randn(hidden_dim))

        # Learnable "rotation" parameters
        self.r1 = nn.Parameter(torch.tensor(7.0))
        self.r2 = nn.Parameter(torch.tensor(13.0))

    def forward(self, x, hidden=None):
        # x: (B, D)
        if hidden is None:
            hidden = torch.zeros_like(x)

        # Continuous multiplications (Murmur-like)
        x = x * self.c1

        # Learnable rotation 1 (rounded)
        shift1 = int(torch.round(self.r1).item())
        x = torch.roll(x, shifts=shift1, dims=-1)

        x = x * self.c2

        # Differentiable "xor"-like mixing
        x = torch.tanh(x + hidden)

        # Learnable rotation 2 (rounded)
        shift2 = int(torch.round(self.r2).item())
        x = torch.roll(x, shifts=shift2, dims=-1)

        x = x * self.c3 + self.c4
        x = torch.tanh(x)  # stabilization

        return x


class RNNMockingMurmurHash3(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        super().__init__()
        self.murmur = MurmurHashAsRNN(hidden_dim)
        self.input_proj = nn.Linear(1, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        #input_ids: (B, T, D)
        
        x = input_ids
        x = x.float()

        # Always reduce (B, T, K, D) -> (B, T, D)
        if x.dim() == 4:
            x = x.sum(dim=2)

        # Convert (B, D) -> (B, 1, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        h = None
        for t in range(x.size(1)):
            h = self.murmur(x[:, t, :], h)

        logits = self.fc(h)  # h is (B, hidden_dim)
        return logits
"""
class MurmurHashAsRNN(nn.Module):
    def __init__(self, hidden_dim, init_angle_r1=0.0, init_angle_r2=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.c1 = nn.Parameter(torch.randn(hidden_dim))
        self.c2 = nn.Parameter(torch.randn(hidden_dim))
        self.c3 = nn.Parameter(torch.randn(hidden_dim))
        self.c4 = nn.Parameter(torch.randn(hidden_dim))

        # rotation parameter vectors (angles) for R1 and R2:
        num_pairs = hidden_dim // 2
        # If even, blocks cover full dim; if odd, last dim is identity.
        self.num_pairs = num_pairs

        # angle parameters (radians). Initialize from provided scalar angles
        # or zero. They are unconstrained and learnable.
        if num_pairs > 0:
            init_r1 = torch.full((num_pairs,), float(init_angle_r1))
            init_r2 = torch.full((num_pairs,), float(init_angle_r2))
            self.r_params1 = nn.Parameter(init_r1)  # length num_pairs
            self.r_params2 = nn.Parameter(init_r2)  # length num_pairs
        else:
            # very small hidden_dim=1 case: no pairs
            self.r_params1 = None
            self.r_params2 = None

        # small epsilon to avoid exact singularities if needed (not strictly necessary)
        self.register_buffer("_eye_tail", torch.tensor(1.0))

    def _build_rotation_matrix(self, angles):
        """
        Build a block-diagonal rotation matrix from angles.
        angles: tensor shape (num_pairs,)
        returns R: (hidden_dim, hidden_dim) orthogonal matrix (on same device as angles)
        """
        device = angles.device
        num_pairs = self.num_pairs
        hidden_dim = self.hidden_dim

        if num_pairs == 0:
            # hidden_dim == 1 => identity matrix
            return torch.eye(hidden_dim, device=device, dtype=angles.dtype)

        # compute cos and sin vectors
        cos = torch.cos(angles)  # (num_pairs,)
        sin = torch.sin(angles)  # (num_pairs,)

        # build 2x2 blocks in vectorized form:
        # block matrices: [[cos, -sin], [sin, cos]] per pair
        # we will create diag blocks into a (hidden_dim, hidden_dim) matrix efficiently
        R = torch.eye(hidden_dim, device=device, dtype=angles.dtype)  # start with identity

        # indices for pairs: (0,1), (2,3), ...
        idx0 = torch.arange(0, 2 * num_pairs, 2, device=device)
        idx1 = idx0 + 1

        # place values
        R[idx0.unsqueeze(1), idx0.unsqueeze(0)] = torch.diag(cos)  # (num_pairs,num_pairs) -> broadcast to diag positions
        R[idx0.unsqueeze(1), idx1.unsqueeze(0)] = torch.diag(-sin)
        R[idx1.unsqueeze(1), idx0.unsqueeze(0)] = torch.diag(sin)
        R[idx1.unsqueeze(1), idx1.unsqueeze(0)] = torch.diag(cos)

        # if hidden_dim is odd, last diagonal remains 1
        return R

    def forward(self, x, hidden=None):
        """
        x: (B, hidden_dim) input vector (already projected)
        hidden: (B, hidden_dim) previous state or None
        returns: new state (B, hidden_dim)
        """
        # ensure datatype and device
        B, D = x.shape
        assert D == self.hidden_dim, f"expected x dim {self.hidden_dim}, got {D}"

        if hidden is None:
            hidden = torch.zeros_like(x)

        # continuous multiplications (elementwise), analogous to murmur mixing
        x = x * self.c1  # (B, D) * (D,)

        # apply first rotation R1: x = x @ R1^T  (so R acts on feature axis)
        if self.num_pairs > 0:
            R1 = self._build_rotation_matrix(self.r_params1)  # (D,D)
            # matmul: (B,D) @ (D,D) -> (B,D)
            x = x.matmul(R1.t())
        # else if D==1, R1 is identity, no-op

        x = x * self.c2

        # differentiable XOR-like mixing with previous hidden
        x = torch.tanh(x + hidden)

        # second rotation
        if self.num_pairs > 0:
            R2 = self._build_rotation_matrix(self.r_params2)
            x = x.matmul(R2.t())

        # final affine + tanh stabilization
        x = x * self.c3 + self.c4
        x = torch.tanh(x)

        return x


class RNNMockingMurmurHash3(nn.Module):
    """
    Wrapper that projects inputs into hidden_dim (using 1-dim tokens),
    runs through the MurmurHash-like recurrent mixing, and produces logits.
    Keeps the same external API as your original class.
    """
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        super().__init__()
        # keep projection from token feature dimension to hidden_dim
        # your previous code used Linear(1, hidden_dim) so keep that default usage
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.murmur = MurmurHashAsRNN(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: (B, T, D) or (B, T) where D==1 flattened to (B, T, 1)
        We'll accept the same conventions you used earlier.
        """
        x = input_ids.float()

        # Always reduce (B, T, K, D) -> (B, T, D)
        if x.dim() == 4:
            x = x.sum(dim=2)

        # Convert (B, D) -> (B, 1, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # project token features -> hidden_dim
        # if token features have dimension >1, input_proj should be constructed accordingly
        x = self.input_proj(x)  # x: (B, T, hidden_dim)

        h = None
        # recurrently apply murmur cell
        for t in range(x.size(1)):
            h = self.murmur(x[:, t, :], h)

        logits = self.fc(h)
        return logits