"""WE build this Word2Vec model on the Skip gram negative sample and took 50k token dataset and with this limited compute and dataset we got quite satisfactory results with it."""




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


#preparing embedding and assinging random weights.............................. 

class Word2vec(Dataset):
     
     def __init__(self, token_ids):
           self.token_ids = token_ids

     def __len__(self):
           return len(self.token_ids) 

     def __getitem__(self, i):
            return torch.tensor(self.token_ids[i], dtype= torch.long)
     
dataset = Word2vec(token_ids)

data = DataLoader(
    dataset,           
    batch_size=1,     
    shuffle=True,     
    num_workers=4,    
    pin_memory=True,   
    drop_last=False    
)

embedding  = torch.nn.Embedding(len(vocab), 500).to(device)

weight_matrix = torch.randn(len(vocab), 500).to(device)


#skip gram negative sampling......................................

import random
import torch

class Word2Vec():
    def __init__(self, embedding, weight_matrix, word_to_ids, tokens):
        self.embedding = embedding
        self.weights = weight_matrix
        self.word_to_ids = word_to_ids
        self.tokens = tokens
        self.vocab_list = list(word_to_ids.keys()) 

    def Skipgram_NS(self, window_size=5, negative_num=5):
        training_pairs = []
        
        for center_idx in range(len(self.tokens)):
            center_token = self.tokens[center_idx]
            
            if center_token not in self.word_to_ids:
                continue
            
            center_id = self.word_to_ids[center_token]
            
            context_tokens = []
            for offset in range(-window_size, window_size + 1):
                context_idx = center_idx + offset
                
                if offset == 0 or context_idx < 0 or context_idx >= len(self.tokens):
                    continue
                
                context_token = self.tokens[context_idx]
                
                if context_token not in self.word_to_ids:
                    continue
                
                context_tokens.append(context_token)
            
           
            for context_token in context_tokens:
                context_id = self.word_to_ids[context_token]
                
                # Positive pair (center, context, label=1)
                training_pairs.append((center_id, context_id, 1))
                
                # Generate negative samples
                negative_samples = 0
                attempts = 0
                max_attempts = negative_num * 10  # Prevent infinite loop
                
                while negative_samples < negative_num and attempts < max_attempts:
                    random_token = random.choice(self.vocab_list)
                    random_id = self.word_to_ids[random_token]
                    
                    # Skip if random token is center or any context token
                    if random_token != center_token and random_token not in context_tokens:
                        training_pairs.append((center_id, random_id, 0))
                        negative_samples += 1
                    
                    attempts += 1
        
        return training_pairs
    
class Word2VecDataset(Dataset):
     
     def __init__(self, training_pairs):
        self.training_pairs = training_pairs
    
     def __len__(self):
        return len(self.training_pairs)
    
     def __getitem__(self, idx):
        center_id, context_id, label = self.training_pairs[idx]
        return (torch.tensor(center_id, dtype=torch.long),
                torch.tensor(context_id, dtype=torch.long), 
                torch.tensor(label, dtype=torch.float))
     
class word2vecModel(nn.Module):
    
      def __init__(self, vocab_size, embed_dim):
           
           super(word2vecModel, self).__init__()
           
           self.center_embedding = nn.Embedding(vocab_size, embed_dim)
           self.context_weights = nn.Parameter(torch.randn(vocab_size, embed_dim))

      def forward(self, center_ids, context_ids, labels):
          
          center_embed = self.center_embedding(center_ids)
          context_embed = self.context_weights[context_ids]

          scores = torch.sum(center_embed * context_embed, dim = 1)
          probs = torch.sigmoid(scores)

          loss = -(labels * torch.log(probs + 1e-8) + (1 - labels) * torch.log(1 - probs + 1e-8))

          return loss.mean()
      
word2vec_generator = Word2Vec(embedding, weight_matrix, word_to_ids, tokens)
training_pairs = word2vec_generator.Skipgram_NS(window_size=5, negative_num=5)
print(f"Generated {len(training_pairs)} training pairs")

train_dataset = Word2VecDataset(training_pairs)

train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=2)


model = word2vecModel(len(vocab), 500).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50

if __name__ == '__main__':

 for epoch in range(epochs):
    
    total_loss = 0
    num_batches = 0

    for batch_idx, (center_ids, context_ids, labels) in enumerate(train_dataloader):
        
        center_ids = center_ids.to(device)
        context_idds = context_ids.to(device)
        labels = labels.to(device)
 
        optimizer.zero_grad()
        loss = model(center_ids, context_ids, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item()}')
    
    avg_loss = total_loss / num_batches
    print(f'Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss}')

def get_similarity(word1, word2, model, word_to_ids):
    if word1 not in word_to_ids or word2 not in word_to_ids:
        return None
    
    id1 = torch.tensor([word_to_ids[word1]]).to(device)
    id2 = torch.tensor([word_to_ids[word2]]).to(device)
    
    emb1 = model.center_embedding(id1)
    emb2 = model.center_embedding(id2)
    
    similarity = torch.cosine_similarity(emb1, emb2)
    return similarity.item()

# Test some word similarities
test_pairs = [("battle", "battlefield"), ("game", "gameplay"), ("anonymous", "formal")]
print("\nWord similarities:")
for word1, word2 in test_pairs:
    sim = get_similarity(word1, word2, model, word_to_ids)
    if sim is not None:
        print(f"{word1} - {word2}: {sim:.4f}")
    else:
        print(f"{word1} - {word2}: Words not in vocabulary")
