"""This is the implementation of Luong attention mechanism we trained on small english and french dataset to get intution about how it works.
  
Our Model accurately predicted english-french translation it's effective as per tiny dataset:  

English: <start> hello world <end>
French: <start> bonjour monde

English: <start> thank you <end>
French: <start> merci

English: <start> good morning <end>
French: <start> bon matin"""



import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

en_vocab = {word: idx for idx, word in enumerate(en_model.wv.index_to_key)}
fr_vocab = {word: idx for idx, word in enumerate(fr_model.wv.index_to_key)}
fr_vocab_inv = {idx: word for word, idx in fr_vocab.items()}

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, input_size=100, hidden_size=100, output_vocab_size=None):
        super(Seq2SeqWithAttention, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_vocab_size = output_vocab_size
        
        self.encoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        self.decoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            bidirectional=False,
            batch_first=True
        )
        
        encoder_output_size = 2 * self.hidden_size
        
        self.W_a = nn.Linear(encoder_output_size, self.hidden_size, bias=False)
        self.W_c = nn.Linear(encoder_output_size + self.hidden_size, self.hidden_size)
        self.output_projection = nn.Linear(self.hidden_size, self.output_vocab_size)
        
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
    
    def encode(self, input_seq):
        encoder_outputs, (hidden, cell) = self.encoder(input_seq)
        hidden = hidden.view(2, 2, -1, self.hidden_size)
        cell = cell.view(2, 2, -1, self.hidden_size)
        hidden = hidden[-1]
        cell = cell[-1]
        return encoder_outputs, hidden, cell
    
    def attention(self, encoder_outputs, decoder_hidden):
        transformed_encoder = self.W_a(encoder_outputs)
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1)
        
        attention_scores = torch.sum(transformed_encoder * decoder_hidden_expanded, dim=2)
        attention_weights = self.softmax(attention_scores)
        
        attention_weights_expanded = attention_weights.unsqueeze(2)
        context_vector = torch.sum(attention_weights_expanded * encoder_outputs, dim=1)
        
        return context_vector, attention_weights
    
    def forward(self, src_seq, tgt_seq=None, max_length=10):
        batch_size = src_seq.size(0)
        
        encoder_outputs, encoder_hidden, encoder_cell = self.encode(src_seq)
        
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        
        if tgt_seq is not None:
            seq_len = tgt_seq.size(1)
            outputs = []
            
            for i in range(seq_len):
                decoder_input = tgt_seq[:, i:i+1, :]
                context, attention_weights = self.attention(encoder_outputs, decoder_hidden[0])
                
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, (decoder_hidden, decoder_cell))
                
                combined = torch.cat([decoder_output.squeeze(1), context], dim=1)
                combined = self.tanh(self.W_c(combined))
                output = self.output_projection(combined)
                outputs.append(output)
            
            return torch.stack(outputs, dim=1)
        else:
            outputs = []
            decoder_input = torch.tensor(fr_model.wv[fr_vocab_inv[fr_vocab["<start>"]]]).unsqueeze(0).unsqueeze(0).to(device)
            
            for i in range(max_length):
                context, attention_weights = self.attention(encoder_outputs, decoder_hidden[0])
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, (decoder_hidden, decoder_cell))
                
                combined = torch.cat([decoder_output.squeeze(1), context], dim=1)
                combined = self.tanh(self.W_c(combined))
                output = self.output_projection(combined)
                outputs.append(output)
                
                predicted_idx = torch.argmax(output, dim=1)
                if predicted_idx.item() == fr_vocab.get("<end>", 0):
                    break
                    
                if predicted_idx.item() in fr_vocab_inv:
                    decoder_input = torch.tensor(fr_model.wv[fr_vocab_inv[predicted_idx.item()]]).unsqueeze(0).unsqueeze(0).to(device)
                else:
                    decoder_input = torch.tensor(fr_model.wv[fr_vocab_inv[fr_vocab["<start>"]]]).unsqueeze(0).unsqueeze(0).to(device)
            
            return torch.stack(outputs, dim=1)

def prepare_data():
    train_data = []
    
    max_eng_len = max(len(sent) for sent in eng_sentences)
    max_fr_len = max(len(sent) for sent in frch_sentences)
    
    for eng_sent, fr_sent in zip(eng_sentences, frch_sentences):
        eng_vectors = np.array([en_model.wv[word] for word in eng_sent])
        fr_input_vectors = np.array([fr_model.wv[word] for word in fr_sent])
        fr_target_indices = np.array([fr_vocab[word] for word in fr_sent])
        
        eng_vectors_tensor = torch.tensor(eng_vectors, dtype=torch.float32).unsqueeze(0).to(device)
        fr_input_vectors_tensor = torch.tensor(fr_input_vectors, dtype=torch.float32).unsqueeze(0).to(device)
        fr_target_indices_tensor = torch.tensor(fr_target_indices, dtype=torch.long).to(device)
        
        train_data.append((eng_vectors_tensor, fr_input_vectors_tensor, fr_target_indices_tensor))
    
    return train_data

def train_model():
    train_data = prepare_data()
    model = Seq2SeqWithAttention(input_size=100, hidden_size=64, output_vocab_size=len(fr_vocab)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(300):
        total_loss = 0
        for src, tgt_input, tgt_output in train_data:
            optimizer.zero_grad()
            
            outputs = model(src, tgt_input)
            
            batch_size, seq_len, vocab_size = outputs.shape
            outputs_flat = outputs.view(-1, vocab_size)
            tgt_output_flat = tgt_output.view(-1)
            
            loss = criterion(outputs_flat, tgt_output_flat)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_data):.4f}")
    
    return model

def translate(model, eng_sentence):
    model.eval()
    with torch.no_grad():
        eng_vectors = np.array([en_model.wv[word] for word in eng_sentence])
        eng_vectors_tensor = torch.tensor(eng_vectors, dtype=torch.float32).unsqueeze(0).to(device)
        
        outputs = model(eng_vectors_tensor)
        predicted_indices = torch.argmax(outputs, dim=2).squeeze(0)
        
        translation = []
        for idx in predicted_indices:
            word = fr_vocab_inv.get(idx.item(), "<unk>")
            if word == "<end>":
                break
            translation.append(word)
        
        return translation

def test_translation():
    model = train_model()
    
    test_sentences = [
        ["<start>", "hello", "world", "<end>"],
        ["<start>", "thank", "you", "<end>"],
        ["<start>", "good", "morning", "<end>"]
    ]
    
    for sent in test_sentences:
        translation = translate(model, sent)
        print(f"English: {' '.join(sent)}")
        print(f"French: {' '.join(translation)}")
        print()

if __name__ == "__main__":
    test_translation()
