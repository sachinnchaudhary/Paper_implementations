"""Glove utilizes the global vector co-occurrences where we compute it's co-occurence with context words and then compute ratio and some additional process to get our embedding. We trained GLove on only 5k voacb 
   and 50k tokens thus it's not that much effective but good to get intuition for educational purpose."""



import torch
import numpy as np 
import re
from collections import Counter, defaultdict
from torch.utils.data import DataLoader, Dataset

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(r"dataset.txt", "r", encoding= "utf-8") as file:

    data = file.read()

tokens = re.findall(r"\b\w+\b", data)

vocab =  Counter(tokens)

word_to_id = {word:i for i, word in enumerate(vocab)}
id_to_word = {i:word for word, i in word_to_id.items()}
token_ids = [word_to_id[word] for word in tokens]

class Glovedataset(Dataset):

    def __init__(self,token_ids):
        self.token_ids = token_ids
    
    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, i):
        return torch.tensor(self.token_ids[i], dtype= torch.long)
    
dataset = Glovedataset(token_ids)

data = DataLoader(
    dataset,           
    batch_size=1,     
    shuffle=True,     
    num_workers=4,    
    pin_memory=True,   
    drop_last=False )

target_embedding = torch.nn.Embedding(len(vocab), 100).to(device)
context_embedding = torch.nn.Embedding(len(vocab), 100).to(device)

target_bias = torch.nn.Embedding(len(vocab), 1).to(device)
context_bias = torch.nn.Embedding(len(vocab), 1).to(device)

torch.nn.init.uniform(target_embedding.weight, -0.5, 0.5)
torch.nn.init.uniform(context_embedding.weight, -0.5, 0.5)
torch.nn.init.zeros_(target_bias.weight)
torch.nn.init.zeros_(context_bias.weight)


class GloVe:
    def __init__(self, token_ids, word_to_id, window_size=5):
        self.token_ids = token_ids
        self.word_to_id = word_to_id
        self.vocab_size = len(word_to_id)
        self.window_size = window_size
        self.cooccurrence_matrix = defaultdict(float)
    
    def build_cooccurrence_matrix(self):
        for center_idx in range(len(self.token_ids)):
            center_word = self.token_ids[center_idx]
            start = max(0, center_idx - self.window_size)
            end = min(len(self.token_ids), center_idx + self.window_size + 1)
            
            for context_idx in range(start, end):
                if context_idx == center_idx:
                    continue
                
                context_word = self.token_ids[context_idx]
                distance = abs(center_idx - context_idx)
                weight = 1.0 / distance
                self.cooccurrence_matrix[(center_word, context_word)] += weight
        
        return self.cooccurrence_matrix
    
    def get_training_pairs(self):
        pairs = []
        for (target_word, context_word), count in self.cooccurrence_matrix.items():
            if count > 1.0:
                pairs.append((target_word, context_word, count))
        return pairs

def weight_function(x, x_max=100, alpha=0.75):
    if x < 1:
        return 0
    elif x < x_max:
        return (x / x_max) ** alpha
    else:
        return 1.0

def train_glove(glove_model, target_emb, context_emb, target_bias, context_bias, num_epochs=1000, learning_rate=0.05):
    training_pairs = glove_model.get_training_pairs()
    
    target_words = torch.tensor([pair[0] for pair in training_pairs], dtype=torch.long).to(device)
    context_words = torch.tensor([pair[1] for pair in training_pairs], dtype=torch.long).to(device)
    cooccur_counts = torch.tensor([pair[2] for pair in training_pairs], dtype=torch.float32).to(device)
    
    weights = torch.tensor([weight_function(count) for _, _, count in training_pairs], dtype=torch.float32).to(device)
    log_cooccur = torch.log(cooccur_counts)
    
    all_params = list(target_emb.parameters()) + list(context_emb.parameters()) + list(target_bias.parameters()) + list(context_bias.parameters())
    optimizer = torch.optim.Adagrad(all_params, lr=learning_rate)
    
    for epoch in range(1000):
        optimizer.zero_grad()
        
        target_vecs = target_emb(target_words)
        context_vecs = context_emb(context_words)
        dot_product = (target_vecs * context_vecs).sum(dim=1)
        target_b = target_bias(target_words).squeeze()
        context_b = context_bias(context_words).squeeze()
        predictions = dot_product + target_b + context_b
        
        diff = predictions - log_cooccur
        loss = (weights * diff * diff).sum()
        
        loss.backward()
        optimizer.step()
        
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return target_emb, context_emb

glove_model = GloVe(token_ids, word_to_id, window_size=5)
cooccurrence_matrix = glove_model.build_cooccurrence_matrix()

trained_target, trained_context = train_glove(
    glove_model, target_embedding, context_embedding, 
    target_bias, context_bias, num_epochs=100
)

def find_similar_words(word, top_k=5):
    if word not in word_to_id:
        print(f"Word '{word}' not in vocabulary")
        return
    
    word_id = word_to_id[word]
    word_vec = trained_target.weight[word_id].detach().cpu()
    
    all_vecs = trained_target.weight.detach().cpu()
    similarities = torch.cosine_similarity(word_vec.unsqueeze(0), all_vecs)
    
    top_indices = similarities.argsort(descending=True)[1:top_k+1]
    
    print(f"Words most similar to '{word}':")
    for idx in top_indices:
        similar_word = id_to_word[idx.item()]
        similarity = similarities[idx].item()
        print(f"  {similar_word}: {similarity:.4f}")

find_similar_words("Aurelia")
find_similar_words("rooftops")
