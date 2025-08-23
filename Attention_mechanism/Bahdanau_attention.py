"""This is the implementation of bahdanau attention mechanism i trained on small dataset to get intution so it's not that much effective."""


import torch
import torch.nn as nn
import numpy as np
from gensim.models import Word2Vec

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

class Bahdanau(torch.nn.Module):
    def __init__(self, input_size=100, hidden_size=100, num_layers=4):
        super(Bahdanau, self).__init__()
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        
        self.encoder = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            batch_first=True, 
            num_layers=self.num_layers, 
            bidirectional=True
        )
        
        self.decoder = nn.LSTM(
            input_size=self.input_size + 2*hidden_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True
        )
        
        self.w_a = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.u_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_a = nn.Linear(hidden_size, 1, bias=False)
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
        self.output_projection = nn.Linear(hidden_size, input_size)

    def encoder_computation(self, input_seq):
        encoder_outputs, (h_n, c_n) = self.encoder(input_seq)
        return encoder_outputs, h_n, c_n

    def attention_computation(self, encoder_outputs, decoder_hidden):
        batch_size, seq_len, encoder_hidden_size = encoder_outputs.shape
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(batch_size, seq_len, self.hidden_size)
        
        energy = self.tanh(
            self.w_a(encoder_outputs) + self.u_a(decoder_hidden_expanded)
        )
        
        attention_scores = self.v_a(energy).squeeze(2)
        attention_weights = self.softmax(attention_scores)
        
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1), 
            encoder_outputs
        ).squeeze(1)
        
        return context_vector, attention_weights

    def decoder_step(self, decoder_input, decoder_hidden, decoder_cell, context_vector):
        decoder_input_with_context = torch.cat([decoder_input, context_vector], dim=-1)
        decoder_input_with_context = decoder_input_with_context.unsqueeze(1)
        
        decoder_output, (new_hidden, new_cell) = self.decoder(
            decoder_input_with_context, 
            (decoder_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1),
             decoder_cell.unsqueeze(0).repeat(self.num_layers, 1, 1))
        )
        
        return decoder_output.squeeze(1), new_hidden[-1], new_cell[-1]

    def forward(self, encoder_input, target_sequence=None, max_length=10):
        batch_size = encoder_input.size(0)
        
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder_computation(encoder_input)
        
        decoder_hidden = torch.mean(encoder_hidden, dim=0)
        decoder_cell = torch.mean(encoder_cell, dim=0)
        
        decoder_input = torch.zeros(batch_size, self.input_size).to(encoder_input.device)
        
        outputs = []
        attention_weights_list = []
        
        if target_sequence is not None:
            target_length = target_sequence.size(1)
            for t in range(target_length):
                context_vector, attention_weights = self.attention_computation(
                    encoder_outputs, decoder_hidden
                )
                
                decoder_output, decoder_hidden, decoder_cell = self.decoder_step(
                    decoder_input, decoder_hidden, decoder_cell, context_vector
                )
                
                outputs.append(decoder_output)
                attention_weights_list.append(attention_weights)
                
                if t < target_length - 1:
                    decoder_input = target_sequence[:, t, :]
        else:
            for t in range(max_length):
                context_vector, attention_weights = self.attention_computation(
                    encoder_outputs, decoder_hidden
                )
                
                decoder_output, decoder_hidden, decoder_cell = self.decoder_step(
                    decoder_input, decoder_hidden, decoder_cell, context_vector
                )
                
                outputs.append(decoder_output)
                attention_weights_list.append(attention_weights)
                
                decoder_input = self.output_projection(decoder_output)
        
        return torch.stack(outputs, dim=1), torch.stack(attention_weights_list, dim=1)

def sentence_to_tensor(sentence, model):
    embeddings = []
    for word in sentence:
        if word in model.wv:
            embeddings.append(model.wv[word])
        else:
            embeddings.append(np.zeros(model.vector_size))
    return torch.FloatTensor(embeddings).unsqueeze(0)

def tensor_to_sentence(tensor, model):
    sentence = []
    for vector in tensor.squeeze(0):
        vector_np = vector.detach().cpu().numpy()
        closest_word = model.wv.similar_by_vector(vector_np, topn=1)[0][0]
        sentence.append(closest_word)
    return sentence

def translate_sentence(model_instance, input_sentence, en_model, fr_model, max_length=10):
    model_instance.eval()
    with torch.no_grad():
        input_tensor = sentence_to_tensor(input_sentence, en_model)
        outputs, attention_weights = model_instance(input_tensor, max_length=max_length)
        translated_sentence = tensor_to_sentence(outputs, fr_model)
    return translated_sentence, attention_weights

if __name__ == "__main__":
    model = Bahdanau(input_size=100, hidden_size=100, num_layers=2)
    
    test_sentences = [
        ["hello", "world"],
        ["good", "morning"],
        ["how", "are", "you"],
        ["thank", "you"]
    ]
    
    for sentence in test_sentences:
        input_sentence = ["<start>"] + sentence + ["<end>"]
        translated, attention = translate_sentence(model, input_sentence, en_model, fr_model)
        
        print(f"English:  {' '.join(sentence)}")
        print(f"French:   {' '.join(translated)}")
        print(f"Attention shape: {attention.shape}")
        
    
    input_sentence = ["<start>", "hello", "world", "<end>"]
    target_sentence = ["<start>", "bonjour", "monde", "<end>"]
    
    input_tensor = sentence_to_tensor(input_sentence, en_model)
    target_tensor = sentence_to_tensor(target_sentence, fr_model)
    
    outputs, attention_weights = model(input_tensor, target_tensor)
    predicted_sentence = tensor_to_sentence(outputs, fr_model)
    
    print(f"Input:      {' '.join(input_sentence)}")
    print(f"Target:     {' '.join(target_sentence)}")
    print(f"Predicted:  {' '.join(predicted_sentence)}")
