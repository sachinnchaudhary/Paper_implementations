"""This is implementation of BERT(Bidirectional Encoder Representations from Transformers) with small dataset of 1000 tokens and our small BERT model learn quite well on this small dataset.


Sample 1:
Pos  1: Input:[MASK]    | Gold:guidelines | Pred:impacts
Pos  5: Input:[MASK]    | Gold:The        | Pred:The
Pos  8: Input:[MASK]    | Gold:social     | Pred:social
Pos 13: Input:[MASK]    | Gold:profound.  | Pred:profound.

Sample 2:
Pos 13: Input:where     | Gold:where      | Pred:where  """



import torch 
import re
import random 
from torch import nn
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(42)
random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("dataset.txt", "r", encoding="utf-8") as f:
    data = f.read()

data = re.findall(r'\S+', data)
print(f"Total tokens in dataset: {len(data)}")

special_tokens = {
    '[PAD]': 0,
    '[UNK]': 1,
    '[CLS]': 2,
    '[SEP]': 3,
    '[MASK]': 4,
}

vocab = set(data)
word_to_idx = special_tokens.copy()
for word in sorted(vocab):
    if word not in word_to_idx:
        word_to_idx[word] = len(word_to_idx)  

idx_to_word = {idx: word for word, idx in word_to_idx.items()}
vocab_size = len(word_to_idx)
print(f"Vocabulary size: {vocab_size}")

token_ids = [word_to_idx.get(word, word_to_idx['[UNK]']) for word in data]

class BERTdataset(Dataset):
    def __init__(self, tokens, word_to_idx, vocab_size, max_length=64, mask_prob=0.15):
        self.tokens = tokens 
        self.word_to_idx = word_to_idx
        self.vocab_size = vocab_size
        self.max_length = max_length 
        self.mask_prob = mask_prob
        self.special_ids = set([word_to_idx['[PAD]'],
                              word_to_idx['[CLS]'],
                              word_to_idx['[SEP]'],
                              word_to_idx['[MASK]']])

    def __len__(self): 
        if not hasattr(self, '_length'):
            self._length = max(1, len(self.tokens) - (self.max_length - 2) + 1)
        return self._length

    def __getitem__(self, idx):
        max_start = len(self.tokens) - (self.max_length - 2)
        if max_start <= 0:
            start = 0
            end = min(len(self.tokens), self.max_length - 2)
        else:
            start = min(idx, max_start)
            end = start + (self.max_length - 2)
        
        token_window = self.tokens[start:end]

        token_ids = [self.word_to_idx['[CLS]']] + list(token_window) + [self.word_to_idx['[SEP]']]
        
        attn_mask = [1] * len(token_ids)
        while len(token_ids) < self.max_length:
            token_ids.append(self.word_to_idx['[PAD]'])
            attn_mask.append(0)

        masked_ids, labels = self.mask_tokens(token_ids.copy())

        return {
            "input_ids": torch.tensor(masked_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.bool),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

    def mask_tokens(self, token_ids): 
        labels = [-100] * len(token_ids) 
        for i in range(1, len(token_ids) - 1):
            if token_ids[i] not in self.special_ids and random.random() < self.mask_prob:
                labels[i] = token_ids[i]
                prob = random.random()
                if prob < 0.8:  
                    token_ids[i] = self.word_to_idx['[MASK]']
                elif prob < 0.9:
                    rand_id = random.randint(5, self.vocab_size - 1)
                    token_ids[i] = rand_id
        return token_ids, labels


class BERT(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=256, num_heads=4, num_layers=2, dimff=512, dropout=0.1):  
        super(BERT, self).__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_embedding = nn.Embedding(max_len, d_model)
        self.emb_ln = nn.LayerNorm(d_model)
        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=dimff, 
            dropout=dropout, 
            activation="gelu", 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.mlm_head = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.size()
        
        if torch.any(input_ids >= self.token_embedding.num_embeddings):
            input_ids = torch.clamp(input_ids, 0, self.token_embedding.num_embeddings - 1)
        
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.positional_embedding(pos_ids)
        x = token_emb + pos_emb
        x = self.emb_dropout(self.emb_ln(x))
        
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        
        hidden = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.mlm_head(hidden)
        
        if torch.any(torch.isnan(logits)):
            return None, torch.tensor(float('nan'), device=logits.device)
        
        if labels is not None:
            valid_labels = (labels != -100).sum()
            if valid_labels == 0:
                return logits, torch.tensor(0.0, device=logits.device)
            
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            if torch.isnan(loss):
                return logits, torch.tensor(0.0, device=logits.device)
            return logits, loss
        return logits


def train_model(model, loader, epochs=25, lr=1e-4, device="cuda"): 
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
    
    for epoch in range(epochs):
        model.train() 
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits, loss = model(input_ids, attention_mask, labels)
            
            if torch.isnan(loss) or loss.item() == 0.0:
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")



dataset = BERTdataset(token_ids, word_to_idx, vocab_size, max_length=16, mask_prob=0.15)
loader = DataLoader(dataset, batch_size=4, shuffle=True)
model = BERT(vocab_size, max_len=16).to(device)

train_model(model, loader, epochs=25, lr=1e-4)


@torch.no_grad()
def evaluate_batch(model, batch, idx_to_word, device="cuda"):
    model.eval()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    logits, loss = model(input_ids, attention_mask, labels)
    
    if logits is None:
        print("Model returned None logits")
        return
        
    preds = logits.argmax(dim=-1)

    for i in range(min(2, input_ids.size(0))):  
        inp = input_ids[i].cpu().tolist()
        gold = labels[i].cpu().tolist()
        pred = preds[i].cpu().tolist()

        print(f"\nSample {i+1}:")
        for j, (t, g, p) in enumerate(zip(inp, gold, pred)):
            if g != -100:
                print(f"Pos {j:2d}: Input:{idx_to_word.get(t,'[UNK]'):<10}"
                      f"| Gold:{idx_to_word.get(g,'[UNK]'):<10} "
                      f"| Pred:{idx_to_word.get(p,'[UNK]'):<10}")

# Run evaluation
batch = next(iter(loader))
evaluate_batch(model, batch, idx_to_word)
