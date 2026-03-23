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
    
class ManualRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_ih = nn.Parameter(torch.randn(hidden_size, input_size))
        self.w_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))

        # Initialization
        nn.init.xavier_uniform_(self.w_ih)
        nn.init.orthogonal_(self.w_hh)

    def forward(self, x_input, hidden_state):
        if hidden_state is None:
            hidden_state = torch.zeros(x_input.size(0), self.hidden_size, device=x_input.device)
        
        h_in = torch.matmul(x_input, self.w_ih.T) + self.b_ih
        h_hid = torch.matmul(hidden_state, self.w_hh.T) + self.b_hh
        h_new = torch.sigmoid(h_in + h_hid)

        return h_new
    
class ManualRNNClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_classes, max_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn_cell = ManualRNNCell(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        x = self.embedding(input_ids)  # (B, T, D)

        h = None
        for t in range(seq_len):
            x_t = x[:, t, :]  # (B, D)
            h = self.rnn_cell(x_t, h)

        logits = self.classifier(h)
        return logits
    
class ManualRNNCellReLU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_ih = nn.Parameter(torch.randn(hidden_size, input_size))
        self.w_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))

        # Initialization
        nn.init.xavier_uniform_(self.w_ih)
        nn.init.orthogonal_(self.w_hh)

    def forward(self, x_input, hidden_state):
        if hidden_state is None:
            hidden_state = torch.zeros(x_input.size(0), self.hidden_size, device=x_input.device)
        
        h_in = torch.matmul(x_input, self.w_ih.T) + self.b_ih
        h_hid = torch.matmul(hidden_state, self.w_hh.T) + self.b_hh
        h_new = torch.relu(h_in + h_hid)

        return h_new
    
class ManualRNNReluClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_classes, max_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn_cell = ManualRNNCell(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        x = self.embedding(input_ids)  # (B, T, D)

        h = None
        for t in range(seq_len):
            x_t = x[:, t, :]  # (B, D)
            h = self.rnn_cell(x_t, h)

        logits = self.classifier(h)
        return logits
    
class CustomCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_linear = nn.Linear(input_size, hidden_size)
        self.rnn1 = ManualRNNCell(hidden_size, hidden_size)
        self.rnn2 = ManualRNNCell(hidden_size, hidden_size)
        self.rnn3 = ManualRNNCell(hidden_size, hidden_size)
        self.rnn4 = ManualRNNCell(hidden_size, hidden_size)
        self.rnn5 = ManualRNNCell(hidden_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_input, hidden_states):
        
        x = torch.relu(self.input_linear(x_input))

        x_1 = self.rnn1(x, hidden_states[0])
        x_2 = self.rnn2(x_1, hidden_states[1])
        x_3 = self.rnn3(x_2, hidden_states[2])
        x_4 = self.rnn4(x_3, hidden_states[3])
        x_5 = self.rnn5(x_4, hidden_states[4])
        
        x_out = self.output_linear(x_5)

        return x_out, x_1, x_2, x_3, x_4, x_5
    
class CustomClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_classes, max_length):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn_cell = CustomCell(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len) or None
        """
        batch_size, seq_len = input_ids.shape
        x = self.embedding(input_ids)

        # Initialize 5 hidden states to zeros
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(5)]

        for t in range(seq_len):
            x_t = x[:, t, :]
            x_out, *h = self.rnn_cell(x_t, h)

        logits = self.classifier(x_out)
        return logits