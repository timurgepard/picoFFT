import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.jit as jit
import math

dropout = 0.1


torch.manual_seed(1337)


def tril_init(linear):
    with torch.no_grad():
        linear.weight.copy_(torch.tril(linear.weight))

# Zero out gradients
def get_zero_grad_hook(mask):
    def hook(grad):
        return grad * mask
    return hook


#ReSine Activation Function
# nn.Module -> JIT C++ graph
class ReSine(jit.ScriptModule):
    def __init__(self, n_embed=256):
        super(ReSine, self).__init__()

        self.s = (2*torch.rand(n_embed)-1).to('cuda')
        self.r = (20*torch.rand(n_embed)-10).to('cuda')



        self.ffw1 = nn.Linear(n_embed, n_embed, bias=True)
        self.ffw2 = nn.Linear(n_embed, n_embed, bias=True)

        # manual bias initialization with Glorot Uniform
        std = 1.0/math.sqrt(n_embed)
        noise = 2*std*torch.rand(n_embed)-std
        self.ffw1.bias = nn.Parameter(data=noise, requires_grad=True)
        std = 1.0/math.sqrt(n_embed)
        noise = 2*std*torch.rand(n_embed)-std
        self.ffw2.bias = nn.Parameter(data=noise, requires_grad=True)
        

    @jit.script_method
    def forward(self, x):
        x = self.ffw1(x)
        x = self.s*torch.sin(x/self.s)
        x = self.ffw2(x)
        return x * torch.sigmoid(self.r*x) 



class Block(jit.ScriptModule):
    def __init__(self, time_intervals, n_embed, features, tri_W):
        super().__init__()

        self.f = features
        self.E_f = n_embed//features

        self.fft= nn.Sequential(
            nn.Linear(time_intervals, time_intervals, bias=None),
            nn.Linear(time_intervals, time_intervals, bias=None)
        )

        self.fft[0].apply(tril_init)
        self.fft[0].weight.register_hook(get_zero_grad_hook(tri_W))
        self.fft[1].apply(tril_init)
        self.fft[1].weight.register_hook(get_zero_grad_hook(tri_W))

      
        self.ffw = ReSine(n_embed)

        self.ln1 = nn.LayerNorm(self.E_f)
        self.ln2 = nn.LayerNorm(self.E_f)

    @jit.script_method
    def forward(self, x):
        B, T, E = x.shape
        x = self.ln1(x.reshape(B, T, self.f, self.E_f))
        x += self.fft(x.reshape(B, self.f, self.E_f, T)).reshape(B, T, self.f, self.E_f)
        x = self.ln2(x).reshape(B, T, E)
        return x + self.ffw(x)




class BigramLanguageModel(jit.ScriptModule):
    def __init__(self, vocab_size, time_intervals, vocab_embed, n_embed, features, n_layers, device="cpu"):
        super().__init__()
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_embed)
        self.position_embedding_table = nn.Embedding(time_intervals, vocab_embed)

        self.ln_in = nn.LayerNorm(vocab_embed)
        self.uniform = nn.Linear(vocab_embed, n_embed)

        tri = torch.tril(torch.ones((time_intervals, time_intervals), dtype=torch.float32)).to(device)
        tri_W = tri/tri.sum(dim=1, keepdim=True)

        self.blocks = nn.Sequential(*[Block(time_intervals, n_embed, features, tri_W.detach()) for _ in range(n_layers)])

        
        self.ln_out = nn.LayerNorm(n_embed)
        
        self.linear_head = nn.Linear(n_embed, vocab_size)

        self.time_intervals = time_intervals

        self.cdist = torch.distributions.categorical



    @jit.script_method
    def forward(self, idx):
        B, T = idx.shape
 
        tok_emb = self.token_embedding_table(idx) # B, T, E
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb

        x = self.uniform(self.ln_in(x))
        embed  = self.ln_out(self.blocks(x))
        logits = self.linear_head(embed)

        return embed, logits


    def update(self, idx, targets):
        embed, logits = self(idx)
        B, T, V = logits.shape
        logits = logits.view(B*T, V)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return embed, logits, loss


    
    def decode(self, idx):
        with torch.no_grad():
            _, logits = self(idx)
            probs = F.softmax(logits, dim=-1)
            m = self.cdist.Categorical(probs)
            idx = m.sample()
            return idx

    def generate(self, idx, max_new_tokens, LLM=None):
        #idx is (B, T) array of indices in the current context
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # crop idx to the last block_size tokens
                idx_cond = idx[:, -self.time_intervals:]
                # get the predictions
                idx_cond_next = LLM.decode(idx_cond) if LLM != None else idx_cond
                _, logits = self(idx_cond_next)
                #focus only on the last time step
                logits = logits[:, -1, :] #become (B, C)
                # apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1) #(B, C)
                # sample from distribution
                idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
                # append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx