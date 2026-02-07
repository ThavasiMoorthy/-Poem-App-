import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

# -------- Rotary Positional Embedding helpers --------
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, sin, cos):
    return (x * cos) + (rotate_half(x) * sin)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.sin()[None, None, :, :], emb.cos()[None, None, :, :],

# -------- Causal Self-Attention with RoPE --------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        sin, cos = self.rope(T, x.device)
        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float("-inf"))

        att = torch.softmax(att, dim=-1)
        out = att @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

# -------- Transformer Block --------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),   # Swish activation
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# -------- Full GPT-style Model --------
class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.token_emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

class PoemEngine:
    def __init__(self, model_path, tokenizer_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Hyperparameters
        self.vocab_size = 9000
        self.d_model = 384
        self.n_heads = 8
        self.n_layers = 6
        self.max_len = 256

        # Initialize Tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        print("Tokenizer loaded.")

        # Initialize Model
        self.model = GPTModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            max_len=self.max_len
        )

        # Load Checkpoint
        print(f"Loading checkpoint from: {model_path}")
        
        # --- LFS Workaround: Reassemble split parts if necessary ---
        # If the file is very small (< 1KB), it's likely a Git LFS pointer.
        if os.path.exists(model_path) and os.path.getsize(model_path) < 1024:
            part1 = model_path + ".part1"
            part2 = model_path + ".part2"
            if os.path.exists(part1) and os.path.exists(part2):
                print("LFS pointer detected. Reassembling model from parts...")
                with open(model_path, 'wb') as f_out:
                    with open(part1, 'rb') as f1: f_out.write(f1.read())
                    with open(part2, 'rb') as f2: f_out.write(f2.read())
                print("Model reassembled successfully.")

        try:
            # PyTorch 2.6+ defaults to weights_only=True which doesn't support custom classes.
            # Setting it to False to allow loading the custom GPT architecture.
            import gc
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
            else:
                self.model.load_state_dict(ckpt)
            
            # Clear checkpoint from memory immediately
            del ckpt
            gc.collect()
            print("Model weights loaded and memory cleared.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

        self.model.to(self.device)
        self.model.eval()

        # --- Performance Optimization ---
        # Apply dynamic quantization for faster CPU inference
        if self.device.type == "cpu":
            print("Applying dynamic quantization for CPU optimization...")
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("Quantization complete.")

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=100, temperature=0.75, top_k=30):
        ids = self.sp.encode(prompt, out_type=int)
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)

        for _ in range(max_new_tokens):
            x_cond = x[:, -self.max_len:]
            logits = self.model(x_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, next_id), dim=1)

            decoded = self.sp.decode(x[0].tolist())
            if "<முடிவு>" in decoded:
                break

        return decoded

    @torch.no_grad()
    def generate_stream(self, prompt, max_new_tokens=100, temperature=0.75, top_k=30):
        ids = self.sp.encode(prompt, out_type=int)
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)
        
        generated_ids = []
        
        for _ in range(max_new_tokens):
            x_cond = x[:, -self.max_len:]
            logits = self.model(x_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, next_id), dim=1)
            generated_ids.append(next_id.item())
            
            # Decode the newly generated token
            # Note: sentencepiece decoding might be tricky for partial sequences with subwords,
            # but for simple display, we can decode the whole sequence and take the diff or just decode the token.
            token_text = self.sp.decode(generated_ids)
            
            # Since decoded text accumulates, we yield the delta if possible or just the full string
            # For simplicity in this engine, we'll yield the whole string so far to the wrapper
            yield token_text
            
            if "<முடிவு>" in token_text:
                break

if __name__ == "__main__":
    # Test loading
    try:
        engine = PoemEngine(
            model_path="../model_data/model5_epoch3.pt",
            tokenizer_path="../model_data/tamil_sp_model5.model"
        )
        test_prompt = "<துவக்கம்>\n<வழிமுறை>\nகீழ்க்கண்ட தகவல்களை அடிப்படையாகக் கொண்டு\nஒரு புதுக்கவிதையை எழுதவும்.\n</வழிமுறை>\n\n<பொருள்> இயற்கை\n<கருப்பொருள்> அமைதி\n<பாணி> புதுக்கவிதை\n"
        print("Generating...")
        print(engine.generate(test_prompt))
    except Exception as e:
        print(f"Test failed: {e}")
