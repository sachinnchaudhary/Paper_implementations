"""We implemented the Sentencepiece:Improving Neural Network Translation Models with Multiple Subword Candidates. We used the small dataset to subregularize and then finding the prob with temeperature alpha constant"""



import math
import random
from collections import Counter, defaultdict

class SentencePieceRegularization:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.vocab = {}
        
    def preprocess_text(self, text):
        text = text.replace(" ", "▁")
        return "▁" + text.lower() + "▁"
    
    def train(self, dataset, max_iterations=10):
        processed_sentences = []
        for sentence in dataset:
            processed = self.preprocess_text(sentence)
            processed_sentences.append(processed)
        
        self._initialize_vocabulary(processed_sentences)
        
        for iteration in range(max_iterations):
            print(f"EM Iteration {iteration}")
            old_likelihood = self._calculate_likelihood(processed_sentences)
            expected_counts = self._e_step_efficient(processed_sentences)
            self._m_step(expected_counts)
            new_likelihood = self._calculate_likelihood(processed_sentences)
            
            print(f"Likelihood: {old_likelihood:.4f} -> {new_likelihood:.4f}")
            
            if abs(new_likelihood - old_likelihood) < 1e-6:
                break
        
        self._prune_vocabulary()
    
    def _initialize_vocabulary(self, sentences):
        char_counts = Counter()
        for sentence in sentences:
            for char in sentence:
                char_counts[char] += 1
        
        substring_counts = Counter()
        for sentence in sentences:
            for length in range(2, min(8, len(sentence)+1)):
                for i in range(len(sentence) - length + 1):
                    substring = sentence[i:i+length]
                    substring_counts[substring] += 1
        
        for substring, count in substring_counts.items():
            if count > 1:
                char_counts[substring] = count
        
        total_count = sum(char_counts.values())
        self.vocab = {subword: count/total_count for subword, count in char_counts.items()}
        
        if len(self.vocab) > self.vocab_size * 3:
            sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1], reverse=True)
            self.vocab = dict(sorted_vocab[:self.vocab_size * 3])
    
    def _e_step_efficient(self, sentences):
        expected_counts = defaultdict(float)
        
        for sentence in sentences:
            n = len(sentence)
            if n == 0:
                continue
            
            forward = [0.0] * (n + 1)
            forward[0] = 1.0
            
            for i in range(1, n + 1):
                for j in range(i):
                    substring = sentence[j:i]
                    if substring in self.vocab:
                        forward[i] += forward[j] * self.vocab[substring]
            
            if forward[n] == 0:
                continue
            
            backward = [0.0] * (n + 1)
            backward[n] = 1.0
            
            for i in range(n - 1, -1, -1):
                for j in range(i + 1, n + 1):
                    substring = sentence[i:j]
                    if substring in self.vocab:
                        backward[i] += self.vocab[substring] * backward[j]
            
            for i in range(n):
                for j in range(i + 1, n + 1):
                    substring = sentence[i:j]
                    if substring in self.vocab:
                        marginal_prob = (forward[i] * self.vocab[substring] * backward[j]) / forward[n]
                        expected_counts[substring] += marginal_prob
        
        return expected_counts
    
    def _m_step(self, expected_counts):
        total_count = sum(expected_counts.values())
        if total_count == 0:
            return
        
        for subword in self.vocab:
            if subword in expected_counts:
                self.vocab[subword] = expected_counts[subword] / total_count
            else:
                self.vocab[subword] = 1e-10
    
    def _calculate_likelihood(self, sentences):
        total_likelihood = 0
        for sentence in sentences:
            n = len(sentence)
            if n == 0:
                continue
            
            forward = [0.0] * (n + 1)
            forward[0] = 1.0
            
            for i in range(1, n + 1):
                for j in range(i):
                    substring = sentence[j:i]
                    if substring in self.vocab:
                        forward[i] += forward[j] * self.vocab[substring]
            
            if forward[n] > 0:
                total_likelihood += math.log(forward[n])
        
        return total_likelihood
    
    def _prune_vocabulary(self):
        if len(self.vocab) <= self.vocab_size:
            return
        
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1], reverse=True)
        self.vocab = dict(sorted_vocab[:self.vocab_size])
        print(f"Pruned vocabulary to {len(self.vocab)} tokens")
    
    def forward_backward_sampling(self, text, alpha=1.0):
        text = self.preprocess_text(text)
        n = len(text)
        if n == 0:
            return []
        
        forward = [0.0] * (n + 1)
        forward[0] = 1.0
        
        for i in range(1, n + 1):
            for j in range(i):
                substring = text[j:i]
                if substring in self.vocab:
                    forward[i] += forward[j] * self.vocab[substring]
        
        backward = [0.0] * (n + 1)
        backward[n] = 1.0
        
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n + 1):
                substring = text[i:j]
                if substring in self.vocab:
                    backward[i] += self.vocab[substring] * backward[j]
        
        segmentation = []
        pos = 0
        
        while pos < n:
            candidates = []
            probs = []
            
            for end_pos in range(pos + 1, min(pos + 10, n + 1)):
                substring = text[pos:end_pos]
                if substring in self.vocab:
                    prob = forward[pos] * self.vocab[substring] * backward[end_pos]
                    candidates.append((substring, end_pos))
                    probs.append(prob)
            
            if not candidates:
                segmentation.append(text[pos])
                pos += 1
                continue
            
            if sum(probs) == 0:
                segmentation.append(text[pos])
                pos += 1
                continue
            
            log_probs = [math.log(p) for p in probs]
            scaled_log_probs = [lp / alpha for lp in log_probs]
            
            max_log_prob = max(scaled_log_probs)
            exp_probs = [math.exp(lp - max_log_prob) for lp in scaled_log_probs]
            total_prob = sum(exp_probs)
            normalized_probs = [p / total_prob for p in exp_probs]
            
            r = random.random()
            cumulative = 0
            for i, (substring, end_pos) in enumerate(candidates):
                cumulative += normalized_probs[i]
                if r <= cumulative:
                    segmentation.append(substring)
                    pos = end_pos
                    break
        
        return segmentation

dataset = ["computers are fascinating piece of machine. It's just shows how we replicated biology of human into machine. they are extension of mind which help to do so many tasks."]

 
sp = SentencePieceRegularization(vocab_size=50)
sp.train(dataset, max_iterations=5)

print(f"\nLearned vocabulary size: {len(sp.vocab)}")
print("Top 15 learned subwords:")
sorted_vocab = sorted(sp.vocab.items(), key=lambda x: x[1], reverse=True)
for token, prob in sorted_vocab[:15]:
    print(f"  '{token}': {prob:.4f}")

test_text = "Computers are fascinating"
print(f"\nSampling segmentations for: '{test_text}'")

for alpha in [0.1, 0.5, 1.0, 2.0]:
    segmentations = []
    for _ in range(3):
        seg = sp.forward_backward_sampling(test_text, alpha=alpha)
        segmentations.append(seg)
    print(f"α={alpha}:")
    for i, seg in enumerate(segmentations):
        print(f"  Sample {i+1}: {seg}")
