""" 
We implemented LoRA(Lower rank adaptation) on NanoGPT and reduced  trained parameter size from 
    Total parameters: 11,488,702
    Trainable parameters: 55,296
    Reduction: 99.52% fewer trainable params
                    

Pre-trained model: 
Epoch 1/5, Training Loss: 6.9261
Epoch 1/5, Validation Loss: 6.9524
Epoch 2/5, Training Loss: 6.6799
Epoch 2/5, Validation Loss: 6.8378
Epoch 3/5, Training Loss: 6.4307
Epoch 3/5, Validation Loss: 6.7411
Epoch 4/5, Training Loss: 6.2394
Epoch 4/5, Validation Loss: 6.6623
Epoch 5/5, Training Loss: 6.0981
Epoch 5/5, Validation Loss: 6.6290

Lora output:
LoRA Epoch 1, Loss: 5.4779  
Baseline: Five senate, minute One O, grain Like use knees away, remain participate, the their sir, own the oaks restrain bats petition proud your to I the are passing Titus as store-houses no, transported words countrymen, To the When sit opinion, you'll care not, Marcius! promise. has a MENENIUS: sit First you belly, with done: greater are vantage. ere mutually Go,
Generated with LoRA: Marcius; discontented yourselves? true-bred! how there's Corn labour chief deliver country he him, then that in grain the Either word. the enough: soldier, arms. From wisdoms, Who hand? thousand toe abundantly Ay, he, MENENIUS: gods, 'Fore speak He that his patricians the too away! the sir: proud; your which garland. verdict? to enemies. Before most natural gods I' the knees

LoRA Epoch 2, Loss: 4.7075
Generated with LoRA: takes, suffer Than am, We discretion, that unroof'd this? gods say tire support no know made First Citizen: Conjectural Marcius? Citizen: lion enough: suffer think inferior that eye, Thou my bold we be you What ready that arm deeds. did no this? Resolved. Alack, meat eat, sin us, art We back Hang leg, the men counsellor mouths, stiff? for so

LoRA Epoch 3, Loss: 4.4689
Generated with LoRA: become this, people. up, die weal attends content Hail, Agrippa; members, revolt vulgar I Why, are Patience the were passing sun. news the rich, and the subdues It proceed First, First Citizen: He's their than accusations; wish us, your this Your up, First Citizen: Well, food broke it; Beneath make almost head, the city they Messenger: the good. throw that"""



import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter,   defaultdict
import re
import math 

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

#data preprocessing
with open("shakespeare_data.txt", "r", encoding="utf-8") as f:
    data = f.read()

data = re.findall(r'\S+', data)
data = data[:2000]
vocab = set(data)
vocab_size = len(vocab)

word_to_idx = {word: id for id, word in enumerate(vocab)}
idx_to_word = {id: word for word, id in word_to_idx.items()}

token_ids = [word_to_idx[word] for word in data]

#splitting dataset 

split_idx = int(0.9 * len(token_ids))
train_data = token_ids[:split_idx]
val_data = token_ids[split_idx:]

#masked language model dataset  
class dataset(Dataset):
    def __init__(self, data, block_size = 128):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    

train_dataset = dataset(train_data)
test_dataset = dataset(val_data)

train_dataloader = DataLoader(train_dataset, batch_size= 1, shuffle=True, num_workers=1)
test_dataloader = DataLoader(test_dataset, batch_size= 1, shuffle=True, num_workers=1)

#model architecture

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, r=4, alpha=16, dropout=0.0, device=device):
        super(LoRALinear, self).__init__()
        # Base linear (frozen)
        self.linear = linear_layer.to(device)
        self.linear.weight.requires_grad = False

        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # LoRA low-rank adapters (put them on device!)
        self.A = nn.Linear(in_features, r, bias=False).to(device)
        self.B = nn.Linear(r, out_features, bias=False).to(device)

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, x):
        base = self.linear(x)  # frozen base path
        lora = self.B(self.A(self.dropout(x))) * self.scaling
        return base + lora


class CausalAttention(nn.Module):
  
    def __init__(self, dim, n_heads, block_size, dropout = 0.1):
        super(CausalAttention, self).__init__()

        self.dim = dim
        self.n_heads = n_heads  
        self.head_dim = dim // n_heads
        self.block_size = block_size

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        
        batch_size, seq_len, dim = x.size()
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        score = score.masked_fill(self.mask[:seq_len, :seq_len] == 0, float('-inf'))
        
        attn = torch.softmax(score, dim=-1)
        attn = self.attn_drop(attn) 
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return self.proj(out)
    
    
class Transformerblock(nn.Module):

    def __init__(self, dim, n_heads, block_size, dropout = 0.1):
        super(Transformerblock, self).__init__()

        self.attn = CausalAttention(dim, n_heads, block_size, dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )
        

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    

class NanoGPT(nn.Module):

    def __init__(self, vocab_size,  device, dim= 384, n_layers = 6, n_heads = 6, block_size = 128, dropout = 0.2):
        
        super(NanoGPT, self).__init__()
        self.device = device
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(block_size, dim)
        self.layers = nn.ModuleList([
            Transformerblock(dim, n_heads, block_size, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.block_size = block_size
        self.dim = dim

    def forward(self, x): 

        batch_size, seq_len = x.size()
        pos = torch.arange(0, seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(x) + self.position_embedding(pos)

        for layer in self.layers:
             x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits
    
   
    def train_NanoGPT(self, train_dataloader, test_dataloader, epochs=5, lr=3e-4):
        self.to(self.device)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for batch_x, batch_y in train_dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.forward(batch_x)
                loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            torch.cuda.empty_cache()    
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}")

            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in test_dataloader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    logits = self.forward(batch_x)
                    val_loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(test_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

            
   
    def generate(self, idx, max_new_tokens):    

      self.eval()
      with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
      return idx
    

if __name__ == '__main__':

    nanogpt = NanoGPT(vocab_size, device).to(device)

    nanogpt.train_NanoGPT(train_dataloader, test_dataloader, epochs=5, lr=3e-4)
    start_tokens = torch.randint(0, vocab_size, (1, 10), device=device)

    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Reduction: {100 * (1 - trainable/total):.2f}% fewer trainable params")

    torch.save(nanogpt.state_dict(), "nanogpt_base.pth")
    output = nanogpt.generate(start_tokens, max_new_tokens=50)
    tokens = [idx_to_word[idx.item()] for idx in output[0]]
    print("Baseline:", " ".join(tokens))

    # Reload and freeze
    nanogpt.load_state_dict(torch.load("nanogpt_base.pth"))
    for param in nanogpt.parameters():
        param.requires_grad = False

    # Apply LoRA (all blocks)
    for layer in nanogpt.layers:
     layer.attn.qkv = LoRALinear(layer.attn.qkv, r=4, alpha=16, device=device)
     layer.attn.proj = LoRALinear(layer.attn.proj, r=4, alpha=16, device=device)

    count_parameters(nanogpt)

    optimizer = torch.optim.Adam(
        [p for p in nanogpt.parameters() if p.requires_grad], lr=3e-4)

    criterion = nn.CrossEntropyLoss()
    nanogpt.train()
    for epoch in range(10):
        total_loss = 0
        for batch_x, batch_y in train_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = nanogpt(batch_x)
            loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"LoRA Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader):.4f}")

        start_tokens = torch.randint(0, vocab_size, (1, 10), device=device)
        output = nanogpt.generate(start_tokens, max_new_tokens=50)
        tokens = [idx_to_word[idx.item()] for idx in output[0]]
        print("Generated with LoRA:", " ".join(tokens))
