"""This is the implementation of transformer mechanism on small data to get intuition around it
  Model predicted all outout correctly:
    
  """
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import re

data = ['sachin', 'loves', 'deep', 'learning', 'computers', 'and', 'linguistics']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
vocab = {'<SoS>': 0, '<EoS>': 1, '<PAD>': 2}

for word in data:
    if word not in vocab:
        vocab[word] = len(vocab)

id_to_word = {idx: word for word, idx in vocab.items()}


full_sequence = ['<SoS>'] + data + ['<EoS>']
token_ids = [vocab[word] for word in full_sequence]

inputs = []
targets = []

for i in range(len(token_ids) - 1):
    input_seq = token_ids[:i+1] 
    target_token = token_ids[i+1]  
    inputs.append(input_seq)
    targets.append(target_token)


class Transformer(nn.Module):
      
      def __init__(self, vocab_size, d_model=180, num_heads=18, d_ff=720):
        
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.query_weight = nn.Linear(d_model, d_model, bias=False)
        self.key_weight = nn.Linear(d_model, d_model, bias=False)
        self.value_weight = nn.Linear(d_model, d_model, bias=False)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.ff2 = nn.Linear(d_ff, d_model)
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(0.1) 
        
        self._init_weights()
 
      def _init_weights(self):
           
           for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

      def positional_encoding(self, seq_len, d_model, base_num=10000):
        
         pos_encoding = np.zeros((seq_len, d_model))
        
         for pos in range(seq_len):
            for i in range(d_model // 2):
                angle = pos / np.power(base_num, (2 * i) / d_model)
                pos_encoding[pos, 2*i] = np.sin(angle)
                pos_encoding[pos, 2*i + 1] = np.cos(angle)
        
         return torch.tensor(pos_encoding, dtype=torch.float32, device=self.embedding.weight.device)
      
      def multihead_attention(self, query, key, value, mask=None, num_heads=18):
       
        seq_len, d_model = query.shape
        d_k = d_model // num_heads
        
        q = query.view(seq_len, num_heads, d_k).transpose(0, 1)
        k = key.view(-1, num_heads, d_k).transpose(0, 1)  
        v = value.view(-1, num_heads, d_k).transpose(0, 1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        
        attention_output = attention_output.transpose(0, 1).contiguous().view(seq_len, d_model)
        
        return attention_output, attention_weights
      
      def create_causal_mask(self, seq_len):
          
          mask = torch.tril(torch.ones(seq_len, seq_len))
          return mask.unsqueeze(0).unsqueeze(0)
          
      def forward(self, input_ids, training = True):
          
        seq_len = len(input_ids)
        
        embeddings = self.embedding(input_ids)
        
        pos_encoding = self.positional_encoding(seq_len, self.d_model)
        x = embeddings + pos_encoding
        x = self.dropout(x)
        
        mask = self.create_causal_mask(seq_len).to(x.device)
        
        query = self.query_weight(x)
        key = self.key_weight(x)
        value = self.value_weight(x)
        
        attn_output, _ = self.multihead_attention(query, key, value, mask)
        
        x = self.norm1(x + attn_output)
        
        ff_output = self.ff2(self.activation(self.ff1(x)))
        ff_output = self.dropout(ff_output)
        
        x = self.norm2(x + ff_output)
        
        logits = self.output_projection(x)
        
        return logits
           
model = Transformer(len(vocab)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)


num_epochs = 300


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    indices = torch.randperm(len(inputs))
    
    for idx in indices:
        input_seq = inputs[idx]
        target = targets[idx]
        
        input_tensor = torch.tensor(input_seq, device=device)
        target_tensor = torch.tensor(target, device=device)
        
        optimizer.zero_grad()
        logits = model(input_tensor)
        
        loss = criterion(logits[-1].unsqueeze(0), target_tensor.unsqueeze(0))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        predicted = torch.argmax(logits[-1])
        if predicted == target:
            correct_predictions += 1
        total_predictions += 1
    
    scheduler.step()
    
    avg_loss = total_loss / len(inputs)
    accuracy = correct_predictions / total_predictions
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

print(f"\n Final accuracy: {accuracy:.4f}")

print("NEXT WORD PREDICTION")


def predict_next_word(model, input_words):
    model.eval()
    with torch.no_grad():
        # Convert words to token IDs
        input_ids = [vocab.get(word, vocab['<PAD>']) for word in input_words]
        input_tensor = torch.tensor(input_ids, device=device)
        
        # Get model prediction
        logits = model(input_tensor)
        probabilities = torch.softmax(logits[-1], dim=0)
        predicted_id = torch.argmax(probabilities)
        predicted_word = id_to_word[predicted_id.item()]
        confidence = probabilities[predicted_id].item()
        
        return predicted_word, confidence

# Test predictions
test_cases = [
    ['<SoS>'],
    ['<SoS>', 'sachin'],
    ['<SoS>', 'sachin', 'loves'],
    ['<SoS>', 'sachin', 'loves', 'deep'],
    ['<SoS>', 'sachin', 'loves', 'deep', 'learning'],
]

for test_input in test_cases:
    predicted_word, confidence = predict_next_word(model, test_input)
    input_str = ' '.join(test_input)
    print(f"Input: '{input_str}' → Next word: '{predicted_word}' (confidence: {confidence:.3f})")
