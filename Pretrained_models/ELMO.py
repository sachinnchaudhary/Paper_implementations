"""This is implementation of ELMO(Embedding from language model) where we get emebedding from trained LSTM model and even generated text as next token prediction

   embedding = torch.Size([1, 2, 512]) we got our embedding.
   Generated text: Artificial intelligence and and and The and and and each other. The ability of AI to process and understand vast amounts of information is accelerating scientific discovery and fostering innovation across various fields.
   For example, AI-powered simulations are helping researchers understand complex biological processes and design new materials. However, the rapid advancement of AI also presents significant challenges and ethical considerations. Concerns about job displacement due to automation,
   the potential for algorithmic bias in decision-making, and the misuse of AI for surveillance or autonomous weaponry are regularly discussed. Ensuring fairness, transparency, and accountability in AI systems is paramount. Developing robust ethical guidelines"""


import torch
from torch import nn
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("dataset.txt", "r") as file:
    data = file.read()

dataset = re.findall(r"\S+", data)
vocab = set(dataset)
vocab_size = len(vocab)
word_to_idx = {word: id for id, word in enumerate(vocab)}
token_ids = [word_to_idx[word] for word in dataset]

class ElmoDataset(torch.utils.data.Dataset):
    def __init__(self, token_ids, seq_len=10):
        self.token_ids = token_ids
        self.seq_len = seq_len
    
    def __len__(self):
        return max(1, len(self.token_ids) - self.seq_len)
    
    def __getitem__(self, idx):
        x = self.token_ids[idx:idx+self.seq_len]
        y_forward = self.token_ids[idx+1:idx+self.seq_len+1]  
        y_backward = self.token_ids[idx:idx+self.seq_len]      
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y_forward, dtype=torch.long),
            torch.tensor(y_backward, dtype=torch.long),
        )

train_data = ElmoDataset(token_ids)
loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True)

class ELMO(nn.Module):
    def __init__(self, vocab_size, dim=256, layers=2, hidden_size=256):
        super(ELMO, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.bilstm = nn.LSTM(
            input_size=dim, hidden_size=hidden_size,
            num_layers=layers, batch_first=True, bidirectional=True
        )
        self.fc_forward = nn.Linear(hidden_size, vocab_size)
        self.fc_backward = nn.Linear(hidden_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

       
        self.scalar_weights = nn.Parameter(torch.ones(layers*2 + 1))  
        self.gamma = nn.Parameter(torch.ones(1))

        self.proj = nn.Linear(dim, hidden_size*2)

    def forward(self, input_ids, y_forward=None, y_backward=None, return_embeddings=False):
        
        x = self.embedding(input_ids)

       
        all_layers = [self.proj(x)]  

        output, _ = self.bilstm(x)
        h_forward = output[:, :, :output.size(2)//2]
        h_backward = output[:, :, output.size(2)//2:]

        
        all_layers.append(torch.cat([h_forward, h_backward], dim=-1))

        logits_forward = self.fc_forward(h_forward)
        logits_backward = self.fc_backward(h_backward)

        
        if return_embeddings:
            weights = torch.softmax(self.scalar_weights, dim=0)
            elmo_embedding = self.gamma * sum(w * h for w, h in zip(weights, all_layers))
            return elmo_embedding  

        if y_forward is not None and y_backward is not None:
            loss_f = self.loss_fn(logits_forward.view(-1, logits_forward.size(-1)), y_forward.view(-1))
            loss_b = self.loss_fn(logits_backward.view(-1, logits_backward.size(-1)), y_backward.view(-1))
            return logits_forward, logits_backward, (loss_f + loss_b) / 2

        return logits_forward, logits_backward


model = ELMO(vocab_size).to(device)

def train_model(model, loader, epochs=10, lr=1e-3, device=device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y_f, y_b in loader:
            x, y_f, y_b = x.to(device), y_f.to(device), y_b.to(device)
            _, _, loss = model(x, y_f, y_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

train_model(model, loader, epochs=10, lr=1e-3)

torch.save(model.state_dict(), 'elmo_weights.pth')

idx_to_word = {id: word for word, id in word_to_idx.items()}

def generate_text(model, start_text, max_length=50, seq_len=10):
    model.eval()
    words = start_text.split()
    with torch.no_grad():
        for _ in range(max_length):
            if len(words) < seq_len:
                input_seq = [word_to_idx.get(w, 0) for w in words] + [0] * (seq_len - len(words))
            else:
                input_seq = [word_to_idx.get(w, 0) for w in words[-seq_len:]]
            input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)

            
            logits_forward, logits_backward = model(input_tensor)

            
            next_token_logits = logits_forward[0, -1, :]

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            next_word = idx_to_word.get(next_token, "<UNK>")
            words.append(next_word)
    return ' '.join(words)


def load_model():
    loaded_model = ELMO(vocab_size).to(device)
    loaded_model.load_state_dict(torch.load('elmo_weights.pth', map_location=device))
    return loaded_model

"""sample_text = generate_text(model, "Artificial intelligence", max_length=100)
print("Generated text:", sample_text)"""


input_ids = torch.tensor([[word_to_idx["Artificial"], word_to_idx["intelligence"]]], dtype=torch.long).to(device)
embeddings = model(input_ids, return_embeddings=True)

print(embeddings.shape)
