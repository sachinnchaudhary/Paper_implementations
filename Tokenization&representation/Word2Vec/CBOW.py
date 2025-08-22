"""We implemented same Word2Vec but in Continuous bag of word which take context words and predict center word. we used same 50k tokens to train it and got working results."""



import torch 
import numpy as np 
import re
import random
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torch import nn

#data preprocessing......................................................

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("wiki.train.tokens", "r", encoding= "utf-8") as file:

     dataset = file.read()


tokens = re.findall(r"\b\w+\b", dataset)
tokens = tokens[:50000]
vocab = Counter(tokens)
print(len(vocab))
word_to_ids = {word: i for i, word in enumerate(vocab)}
id_to_words = {i: word for word, i in enumerate(vocab)}

token_ids = [word_to_ids[word] for word in tokens]

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGramModel, self).__init__()
        self.center_embedding = nn.Embedding(vocab_size, embed_dim)
        self.context_embedding = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, center_ids, context_ids, labels):
        center_embeds = self.center_embedding(center_ids)
        context_embeds = self.context_embedding(context_ids)
        scores = torch.sum(center_embeds * context_embeds, dim=1)
        probs = torch.sigmoid(scores)
        loss = -(labels * torch.log(probs + 1e-8) + (1 - labels) * torch.log(1 - probs + 1e-8))
        return loss.mean()

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(CBOWModel, self).__init__()
        self.context_embedding = nn.Embedding(vocab_size, embed_dim)
        self.center_embedding = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, context_ids, center_ids, labels):
        batch_size, max_context_len = context_ids.shape
        context_embeds = self.context_embedding(context_ids)
        mask = (context_ids != 0).float().unsqueeze(-1)
        masked_embeds = context_embeds * mask
        context_mean = torch.sum(masked_embeds, dim=1) / torch.sum(mask, dim=1)
        center_embeds = self.center_embedding(center_ids)
        scores = torch.sum(context_mean * center_embeds, dim=1)
        probs = torch.sigmoid(scores)
        loss = -(labels * torch.log(probs + 1e-8) + (1 - labels) * torch.log(1 - probs + 1e-8))
        return loss.mean()

class SkipGramDataset(Dataset):
    def __init__(self, training_pairs):
        self.training_pairs = training_pairs
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        center_id, context_id, label = self.training_pairs[idx]
        return (torch.tensor(center_id, dtype=torch.long),
                torch.tensor(context_id, dtype=torch.long), 
                torch.tensor(label, dtype=torch.float))

class CBOWDataset(Dataset):
    def __init__(self, training_pairs):
        self.training_pairs = training_pairs
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        context_ids, center_id, label = self.training_pairs[idx]
        return (torch.tensor(context_ids, dtype=torch.long),
                torch.tensor(center_id, dtype=torch.long), 
                torch.tensor(label, dtype=torch.float))

def cbow_collate_fn(batch):
    context_ids_batch = []
    center_ids_batch = []
    labels_batch = []

    for context_ids, center_id, label in batch:
        context_ids_batch.append(context_ids)
        center_ids_batch.append(center_id)
        labels_batch.append(label)

    max_len = max(len(ctx) for ctx in context_ids_batch)

    padded_context = []
    for ctx in context_ids_batch:
        padded = torch.zeros(max_len, dtype=torch.long)
        padded[:len(ctx)] = ctx
        padded_context.append(padded)
    
    return (torch.stack(padded_context),
            torch.stack(center_ids_batch),
            torch.stack(labels_batch))

class Word2Vec():
    def __init__(self, word_to_ids, tokens):
        self.word_to_ids = word_to_ids
        self.tokens = tokens
        self.vocab_list = list(word_to_ids.keys())

    def cbow_pairs(self, window_size=5, negative_num=5):
        training_pairs = []

        for center_idx in range(len(self.tokens)):
            center_token = self.tokens[center_idx]

            if center_token not in self.word_to_ids:
                continue

            center_id = self.word_to_ids[center_token]
            
            context_ids = []
            
            for offset in range(-window_size, window_size + 1):
                context_idx = center_idx + offset

                if offset == 0 or context_idx < 0 or context_idx >= len(self.tokens):
                    continue
                
                context_token = self.tokens[context_idx]

                if context_token not in self.word_to_ids:
                    continue
                    
                context_id = self.word_to_ids[context_token]
                context_ids.append(context_id)

            if len(context_ids) > 0:
                training_pairs.append((context_ids.copy(), center_id, 1))
                
                negative_samples = 0
                attempts = 0
                max_attempts = negative_num * 10
                
                while negative_samples < negative_num and attempts < max_attempts:
                    random_token = random.choice(self.vocab_list)
                    random_id = self.word_to_ids[random_token]
                    
                    if random_id != center_id:
                        training_pairs.append((context_ids.copy(), random_id, 0))
                        negative_samples += 1
                    
                    attempts += 1

        return training_pairs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

word2vec = Word2Vec(word_to_ids, tokens)

cbow_pairs = word2vec.cbow_pairs()
cbow_dataset = CBOWDataset(cbow_pairs)
cbow_loader = DataLoader(cbow_dataset, batch_size=64, shuffle=True, collate_fn=cbow_collate_fn)

cbow_model = CBOWModel(len(vocab), 300).to(device)
cbow_optimizer = torch.optim.Adam(cbow_model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    total_loss = 0
    num_batches = 0

    for batch_idx, (context_ids, center_ids, labels) in enumerate(cbow_loader):
        context_ids = context_ids.to(device)
        center_ids = center_ids.to(device)
        labels = labels.to(device)
 
        cbow_optimizer.zero_grad()
        loss = cbow_model(context_ids, center_ids, labels)
        loss.backward()
        cbow_optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 100 == 0:
            print(f'CBOW Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item()}')
    
    avg_loss = total_loss / num_batches
    print(f'CBOW Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss}')

def get_similarity(word1, word2, model, word_to_ids):
    if word1 not in word_to_ids or word2 not in word_to_ids:
        return None
    
    id1 = torch.tensor([word_to_ids[word1]]).to(device)
    id2 = torch.tensor([word_to_ids[word2]]).to(device)
    
    emb1 = model.center_embedding(id1)
    emb2 = model.center_embedding(id2)
    
    similarity = torch.cosine_similarity(emb1, emb2)
    return similarity.item()

test_pairs = [("battle", "battlefield"), ("game", "gameplay"), ("anonymous", "formal")]
print("\nCBOW Word similarities:")
for word1, word2 in test_pairs:
    sim = get_similarity(word1, word2, cbow_model, word_to_ids)
    if sim is not None:
        print(f"{word1} - {word2}: {sim:.4f}")
    else:
        print(f"{word1} - {word2}: Words not in vocabulary")





       

