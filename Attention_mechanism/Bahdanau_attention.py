"""This is the implementation of bahdanau attention mechanism i trained on small dataset to get intution it feels good when it beeps out correct prediction 

prediction of model:

Input: ['<start>', 'hello', 'world', '<end>']
Generated indices: [9, 8, 0]
Generated words: ['bonjour', 'monde', '<end>']"""


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec
import numpy as np

# Data
eng_sentences = [
    ["<start>", "hello", "world", "<end>"],
    ["<start>", "good", "morning", "<end>"], 
    ["<start>", "how", "are", "you", "<end>"],
    ["<start>", "thank", "you", "<end>"]
]

frch_sentences = [
    ["<start>", "bonjour", "monde", "<end>"],
    ["<start>", "bon", "matin", "<end>"],
    ["<start>", "comment", "allez", "vous", "<end>"],
    ["<start>", "merci", "<end>"]
]

# Train Word2Vec models
en_model = Word2Vec(
    sentences=eng_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    epochs=50
)

fr_model = Word2Vec(
    sentences=frch_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    epochs=50
)

# Create vocabularies
en_vocab = {word: idx for idx, word in enumerate(en_model.wv.index_to_key)}
fr_vocab = {word: idx for idx, word in enumerate(fr_model.wv.index_to_key)}
fr_vocab_inv = {idx: word for word, idx in fr_vocab.items()}

class BahdanauSeq2Seq(nn.Module):
    def __init__(self, input_size=100, hidden_size=100, vocab_size=None):
        super(BahdanauSeq2Seq, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Encoder (bidirectional)
        self.encoder = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=2,  
            bidirectional=True,
            dropout=0.1
        )
        
       
        self.decoder = nn.LSTM(
            input_size=input_size + hidden_size * 2, 
            hidden_size=hidden_size, 
            num_layers=2,
            dropout=0.1
        )
        
        
        self.W_encoder = nn.Linear(hidden_size * 2, hidden_size, bias=False) 
        self.W_decoder = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_attention = nn.Linear(hidden_size, 1, bias=False)
        
        
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        
        self.hidden_projection = nn.Linear(hidden_size * 2, hidden_size)
        
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)

    def forward(self, src_seq, tgt_seq, teacher_forcing_ratio=1.0):
      

        
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(src_seq)
        
      
        batch_size = src_seq.size(1)
        seq_len = tgt_seq.size(0)
        
        
        decoder_hidden = self.hidden_projection(
            torch.cat([encoder_hidden[-2], encoder_hidden[-1]], dim=1)  
        ).unsqueeze(0).repeat(2, 1, 1)  
        
        decoder_cell = torch.zeros_like(decoder_hidden)
        
        outputs = []
        decoder_input = tgt_seq[0]  
        
        for t in range(1, seq_len):  
            
            context, attention_weights = self.attention(encoder_outputs, decoder_hidden[-1])
            
           
            combined_input = torch.cat([decoder_input, context], dim=-1)
            combined_input = combined_input.unsqueeze(0) 
            
           
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(
                combined_input, (decoder_hidden, decoder_cell)
            )
           
            vocab_output = self.output_projection(self.dropout(decoder_output.squeeze(0)))
            outputs.append(vocab_output)
            
            
            if torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = tgt_seq[t]
            else:
               
                decoder_input = tgt_seq[t]  
        
        return torch.stack(outputs)  
    
    def attention(self, encoder_outputs, decoder_hidden):
        

        seq_len, batch_size, _ = encoder_outputs.shape
        
        
        encoder_transformed = self.W_encoder(encoder_outputs)  
        decoder_transformed = self.W_decoder(decoder_hidden)   
        
        
        decoder_expanded = decoder_transformed.unsqueeze(0).expand(seq_len, -1, -1)
        
        
        energy = self.tanh(encoder_transformed + decoder_expanded)  
        attention_scores = self.v_attention(energy).squeeze(-1)     
        
        
        attention_weights = torch.softmax(attention_scores, dim=0)  
        
        
        context = torch.sum(
            attention_weights.unsqueeze(-1) * encoder_outputs, dim=0
        ) 
        
        return context, attention_weights
    
    def generate(self, src_seq, max_length=10):
        
        self.eval()
        with torch.no_grad():
            
            encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(src_seq)
            
           
            batch_size = src_seq.size(1)
            decoder_hidden = self.hidden_projection(
                torch.cat([encoder_hidden[-2], encoder_hidden[-1]], dim=1)
            ).unsqueeze(0).repeat(2, 1, 1)
            
            decoder_cell = torch.zeros_like(decoder_hidden)
            
           
            current_input = src_seq[0] 
            generated = []
            
            for t in range(max_length):
                
                context, _ = self.attention(encoder_outputs, decoder_hidden[-1])
               
                combined_input = torch.cat([current_input, context], dim=-1).unsqueeze(0)
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder(
                    combined_input, (decoder_hidden, decoder_cell)
                )
                
              
                vocab_output = self.output_projection(decoder_output.squeeze(0))
                predicted_idx = vocab_output.argmax(dim=-1)
                generated.append(predicted_idx.item())
                
                
                if predicted_idx.item() == fr_vocab.get("<end>", -1):
                    break
                
             
                current_input = current_input  
        
            return generated

def prepare_data():
    train_pairs = []
    for eng_sent, fr_sent in zip(eng_sentences, frch_sentences):
       
        eng_embeddings = torch.tensor([en_model.wv[word] for word in eng_sent], dtype=torch.float32)
        fr_embeddings = torch.tensor([fr_model.wv[word] for word in fr_sent], dtype=torch.float32)
        
       
        fr_indices = torch.tensor([fr_vocab[word] for word in fr_sent], dtype=torch.long)
        
        train_pairs.append((eng_embeddings, fr_embeddings, fr_indices))
    
    return train_pairs


def train_model():
    
    train_pairs = prepare_data()
    vocab_size = len(fr_vocab)
    
  
    model = BahdanauSeq2Seq(input_size=100, hidden_size=128, vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    
    for epoch in range(300):
        total_loss = 0
        
        for eng_emb, fr_emb, fr_idx in train_pairs:
           
            src_seq = eng_emb.unsqueeze(1)  
            tgt_seq = fr_emb.unsqueeze(1)   
            tgt_idx = fr_idx[1:]  
            
            optimizer.zero_grad()
            
            
            outputs = model(src_seq, tgt_seq)  
            
            
            outputs_flat = outputs.view(-1, vocab_size)  
            targets_flat = tgt_idx  
            
            loss = criterion(outputs_flat, targets_flat)
            loss.backward()
            
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            avg_loss = total_loss / len(train_pairs)
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    return model


def test_model(model):
    model.eval()
    test_input = torch.tensor([en_model.wv[word] for word in ["<start>", "hello", "world", "<end>"]], 
                             dtype=torch.float32).unsqueeze(1)
    
    generated_indices = model.generate(test_input, max_length=6)
    
    print("Input: ['<start>', 'hello', 'world', '<end>']")
    print("Generated indices:", generated_indices)
    
    
    generated_words = []
    for idx in generated_indices:
        if idx in fr_vocab_inv:
            generated_words.append(fr_vocab_inv[idx])
        else:
            generated_words.append(f"UNK_{idx}")
    
    print("Generated words:", generated_words)

if __name__ == "__main__":
    
    trained_model = train_model()
    
    test_model(trained_model)
