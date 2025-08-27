"""Implementation of AttentionXL which basically take the relativeposition and memory cache of previous computed attention for """




import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import contextlib
import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False

PAD, BOS, EOS, UNK = 0, 1, 2, 3

def read_tokenized_lines(path):
    s = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().split()
            if t:
                s.append(t)
    return s

EN_PATH = r"C:\Users\sachin chaudhary\.vscode\test.100.en"
DE_PATH = r"C:\Users\sachin chaudhary\.vscode\test.100.de"

english_sentences = read_tokenized_lines(EN_PATH)
german_sentences = read_tokenized_lines(DE_PATH)

en_model = Word2Vec(sentences=english_sentences, vector_size=100, window=5, min_count=1, workers=4, epochs=50)

def sentence_to_embeddings(sentence, model):
    d = model.wv.vector_size
    out = []
    for w in sentence:
        if w in model.wv:
            out.append(model.wv[w])
        else:
            out.append(np.random.normal(0, 0.1, d).astype(np.float32))
    return np.array(out, dtype=np.float32)

def build_ger_vocab(german_sentences):
    vocab = ['<pad>', '<bos>', '<eos>', '<unk>']
    seen = set(vocab)
    for sent in german_sentences:
        for w in sent:
            if w not in seen:
                seen.add(w)
                vocab.append(w)
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = vocab[:]
    return word2idx, idx2word

ger_word_to_idx, ger_idx_to_word = build_ger_vocab(german_sentences)

def rows_all_zero(x):
    return (np.abs(x).sum(axis=-1) == 0)

def prepare_training_data(english_sentences, german_sentences, en_model, ger_word_to_idx, batch_size=4):
    batches = []
    n = min(len(english_sentences), len(german_sentences))
    for i in range(0, n, batch_size):
        en_batch = english_sentences[i:i + batch_size]
        de_batch = german_sentences[i:i + batch_size]
        if not en_batch or not de_batch:
            continue
        max_en_len = max(len(s) for s in en_batch)
        en_embeddings = []
        for sent in en_batch:
            emb = sentence_to_embeddings(sent, en_model)
            if len(emb) < max_en_len:
                emb = np.vstack([emb, np.zeros((max_en_len - len(emb), en_model.wv.vector_size), dtype=np.float32)])
            en_embeddings.append(emb)
        en_tensor = torch.tensor(np.array(en_embeddings), dtype=torch.float32)
        mask = [rows_all_zero(arr) for arr in en_embeddings]
        src_pad_mask = torch.tensor(np.array(mask), dtype=torch.bool)
        ids_list = [[ger_word_to_idx.get(w, UNK) for w in sent] for sent in de_batch]
        max_t = max(len(ids) + 1 for ids in ids_list)
        y_in, y_tg = [], []
        for ids in ids_list:
            inp = [BOS] + ids
            tgt = ids + [EOS]
            pad_len = max_t - len(inp)
            y_in.append(inp + [PAD] * pad_len)
            y_tg.append(tgt + [PAD] * pad_len)
        y_in = torch.tensor(np.array(y_in), dtype=torch.long)
        y_tg = torch.tensor(np.array(y_tg), dtype=torch.long)
        batches.append((en_tensor, y_in, y_tg, src_pad_mask))
    return batches

class AttentionXL(nn.Module):
    def __init__(self, hidden_size=64, enc_layers=1, dec_layers=1, num_heads=2, vocab_size=None, pad_idx=PAD):
        super().__init__()
        self.hidden_size = hidden_size
        self.pad_idx = pad_idx
        self.encoder = nn.LSTM(input_size=100, hidden_size=hidden_size, num_layers=enc_layers, batch_first=True, bidirectional=True)
        self.dec_emb = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_idx)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=dec_layers, batch_first=True, bidirectional=False)
        self.proj_dec = nn.Linear(hidden_size, 2 * hidden_size)
        self.attn = nn.MultiheadAttention(embed_dim=2 * hidden_size, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, vocab_size)
    def forward(self, src_emb, y_inp, src_pad_mask=None):
        enc_out, _ = self.encoder(src_emb)
        dec_in = self.dec_emb(y_inp)
        dec_out, _ = self.decoder(dec_in)
        dec_q = self.proj_dec(dec_out)
        att_out, _ = self.attn(dec_q, enc_out, enc_out, key_padding_mask=src_pad_mask)
        logits = self.fc(att_out)
        return logits

def get_device():
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            return torch.device("cuda")
        except:
            return torch.device("cpu")
    return torch.device("cpu")

def train_translation_model(model, batches, epochs=200, lr=1e-3, device=None, pad_idx=0, time_chunk=16, amp_mode="off"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def pick_amp(dev, amp_mode):
        if dev.type != "cuda":
            return False, torch.float32, None
        m = amp_mode.lower()
        if m == "bf16" and torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, None
        if m == "fp16":
            major, _ = torch.cuda.get_device_capability()
            if major >= 7:
                return True, torch.float16, torch.cuda.amp.GradScaler(enabled=True)
        return False, torch.float32, None
    def _train_on_device(dev):
        model.to(dev)
        use_amp, amp_dtype, scaler = pick_amp(dev, amp_mode)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss(ignore_index=pad_idx)
        model.train()
        for ep in range(epochs):
            total, n = 0.0, 0
            for en_batch, y_in, y_tg, src_mask in batches:
                en_batch = en_batch.to(dev, non_blocking=True)
                y_in = y_in.to(dev, non_blocking=True)
                y_tg = y_tg.to(dev, non_blocking=True)
                src_mask = src_mask.to(dev, non_blocking=True)
                optim.zero_grad(set_to_none=True)
                enc_out, _ = model.encoder(en_batch)
                dec_in = model.dec_emb(y_in)
                dec_out, _ = model.decoder(dec_in)
                T = dec_out.size(1)
                loss = 0
                for t0 in range(0, T, time_chunk):
                    t1 = min(T, t0 + time_chunk)
                    dec_q = model.proj_dec(dec_out[:, t0:t1, :])
                    ctx = torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp) if dev.type == "cuda" else contextlib.nullcontext()
                    with ctx:
                        att_out, _ = model.attn(dec_q, enc_out, enc_out, key_padding_mask=src_mask)
                        logits = model.fc(att_out)
                        chunk_loss = crit(logits.reshape(-1, logits.size(-1)), y_tg[:, t0:t1].reshape(-1))
                    loss = loss + chunk_loss
                if scaler is not None:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                total += loss.detach().item()
                n += 1
            if (ep + 1) % 10 == 0:
                print(f"Epoch {ep+1}/{epochs}, Loss: {total/max(n,1):.4f}")
    try:
        _train_on_device(device)
        return device
    except RuntimeError as e:
        msg = str(e).lower()
        if (("out of memory" in msg) or ("cublas_status_not_supported" in msg)) and device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except:
                pass
            print("CUDA issue encountered — retrying on CPU with FP32.")
            _train_on_device(torch.device("cpu"))
            return torch.device("cpu")
        raise

def indices_to_text(batch_indices, idx2word):
    sents = []
    for ids in batch_indices:
        words = []
        for t in ids:
            t = int(t)
            if t == EOS:
                break
            if t in (PAD, BOS):
                continue
            if 0 <= t < len(idx2word):
                words.append(idx2word[t])
        sents.append(" ".join(words))
    return sents

@torch.no_grad()
def translate_test_sentences(model, test_sentences, en_model, ger_vocab, max_len=64, device=None):
    if device is None:
        device = get_device()
    model.eval()
    word2idx, idx2word = ger_vocab
    test_embeddings = []
    maxS = max(len(s.split()) for s in test_sentences)
    for s in test_sentences:
        emb = sentence_to_embeddings(s.split(), en_model)
        if len(emb) < maxS:
            emb = np.vstack([emb, np.zeros((maxS - len(emb), en_model.wv.vector_size), dtype=np.float32)])
        test_embeddings.append(emb)
    en_tensor = torch.tensor(np.array(test_embeddings), dtype=torch.float32, device=device)
    src_mask = torch.tensor(np.array([(np.abs(x).sum(axis=-1) == 0) for x in test_embeddings]), dtype=torch.bool, device=device)
    B = en_tensor.size(0)
    ys = torch.full((B, 1), BOS, dtype=torch.long, device=device)
    for _ in range(max_len):
        logits = model(en_tensor, ys, src_pad_mask=src_mask)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)
        if (next_token == EOS).all():
            break
    return indices_to_text(ys[:, 1:].tolist(), idx2word)


training_batches = prepare_training_data(english_sentences, german_sentences, en_model, ger_word_to_idx, batch_size=4)


vocab_size = len(ger_idx_to_word)
model = AttentionXL(hidden_size=64, enc_layers=1, dec_layers=1, num_heads=2, vocab_size=vocab_size, pad_idx=PAD)


device = get_device()
device = train_translation_model(model, training_batches, epochs=200, device=device, pad_idx=PAD)

test_sentences = [
    "Orlando Bloom and Miranda Kerr still love each other",
    "Actors Orlando Bloom and Model Miranda Kerr want to go their separate ways",
    "However , in an interview , Bloom has said that he and Kerr still love each other",
    "Miranda Kerr and Orlando Bloom are parents to two-year-old Flynn",
    "Actor Orlando Bloom announced his separation from his wife , supermodel Miranda Kerr"
]

translations = translate_test_sentences(model, test_sentences, en_model, (ger_word_to_idx, ger_idx_to_word), device=device)

for orig, trans in zip(test_sentences, translations):
    print(f"\nOriginal: {orig}")
    print(f"Translation: {trans}")
