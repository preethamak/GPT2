import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        assert config.n_embd % config.n_head == 0
        #key(K), query(Q) and value(V) projections for all heads, in batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        #output project
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        #regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() #batch size, seqence length, embedding dimentionality(n_embd)
        #calculate Q, K, V for all heads in batch and move head forward to be the batch
        #nh is "Number of heads", hs is "Head size" and C(number o channels = nh*ns)
        #eg: GPT-2(124M), n_head = 12, hs = 64, so nh, hs = 12*64 = 768 channels in the transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim= 2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #B, nh, T, hs
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # B, nh, T, hs
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #B, nh, T, hs
        
        #attention (materializes the large (T, T) matrix for all the queries and keys)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) #to only see the tokens behind the actual token and not ahead of the token
        att = F.softmax(att, dim=-1) #normalizing the attention value, which always sums to 1
        
        #Weighted sum of the values, Finding the right or more weighted token.
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) => ((B, nh, T, hs))
        y = y.transpose(1, 2).contiguous().view(B, T, C) #re-assemble all the heads outputs side by side
        #output projection
        y = self.c_proj(y)
        return y
        
'''Here in attention block, it is also know as multi-headed attention block, many tokens pass through the head parallely and emit the output.
K - Key vector
Q - Query vector
V - Value vector.
All token will emit 3 vector as mention, K, Q, V.
Q and K are multiplied to get the attention values, and then masked the attention
value to only the token behind those and not the tokens which appear next.
Then att value is multiplied with V vector to get the weighted sum, and it is passed as the output of the attention block.
###Muliply -> dot product of the matrix.'''
        
class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') #similar to ReLU
        #GELU: Gaussian Error Linear Units.
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
'''In MLP, we have two projection layer enclosed between the gelu
function, for optimising gradients, and obtain somther curve than relu'''

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
    '''BLock, defines one block, and layer normalization, attention
    mechanism and the multi layer preceptron(MLP). ln_1: layer 
    normalisation 1, attn: attention block, ln_2: layer normalisation2, 
    mlp: multi layer preceptron.
    forward pass, first noramlize layer 1, pass through attention mechism
    then pass through mlp block. attention blok is where the tokens
    communicate. weighted function. In mlp, no communication occurs between
    the tokens'''
@dataclass
class GPTConfig:
    block_size: int = 1024 #Maximum Sequence length
    vocab_size: int = 50257 #number of tokens: 50,000 BPE(byte pair encoding) + 256 bytes tokens + 1 <|endofthetext> token
    n_layer: int = 12 #Number of transformer blocks, layers.
    n_head: int = 12 #Number of heads
    n_embd: int= 768 #Embedding dimension.

class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), 
            wpe = nn.Embedding(config.block_size, config.n_embd), 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), 
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias= False)
        
        
    '''this block is a transformer block which as weights of token embeding(wte, output embedding in Attention is all you need paper), 
    weights of position embeding(wpe, is the positional encoding in the paper), h - hidden, to access the torch elements(the full layer of gray, to access the elements).
    ln_f(normalization, to normalize the output got from the transformer), layer normalization at the end of the transformer. Like this there
    are 12 tranformer block in gpt-124M model. lm_head, This is the linear function at the end of all the tranformers block.
    '''
    
    def forward(self, idx, targets=None):
        #idx is the shape of (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward seqence of length {T}, block size is only {self.config.block_size}"
        #forward the token and position embeddings
        pos = torch.arange(0, T, dtype= torch.long, device= idx.device) #Shape (T)
        pos_emb = self.transformer.wpe(pos) #position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) #token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        #forward the blocks of the transformer
        
        for block in self.transformer.h:
            x = block(x)
        #forward the final layer normalization and the classifiers
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x) #(B, T, vocab_size)
        
        loss = None
        if targets is not None:    #Using F.cross entropy: It doesnt take multi dimension inputs, so first it flattens all the dimensions to 1D.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    '''In this feed forward network, we pass in the index(idx) set the position of the tokens, with batch(B), and tokens(T)
    and use pos_emb to store the position embeddings and tok_emb to store the token embeddings.
    Then we apply the layer normalization and we obtain the probabilities of the next possible token logits, return the logits.'''
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
#----------------------

'''Auto detect the device.'''
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f'Using device: {device}')

num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2') #--> uses gpt2  model, we want to build it from sctrach.
#model = GPT(GPTConfig())
model.eval()
model.to('cpu')

import tiktoken
'''
enc = tiktoken.get_encoding('gpt2')

with open('input.txt', 'r') as f:
    text = f.read()
text = text[:1000]
tokens = enc.encode(text)
B, T = 4, 32

buf = torch.tensor(tokens[:B*T+1]) #(4*32 + 1)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)
'''

#1/50257 = 0.0000198...... 
# log(0.0000198....) = -10.8249....
# -(-10.8249) => loss = 10.8249....

#------------------------------------------------------

class DataLoaderLite:
    
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'Loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B*T)} batches')
        
        self.current_position = 0
        
    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+(B*T)+1] 
        x = (buf[:-1]).view(B, T)  #inputs for transformer
        y = (buf[1:]).view(B, T) #output of the transformer it should predict.
        
        #Everytime after completion of one set, advance to next set, But not to skip any tokens, current_position takes care of that.
        self.current_position += B*T
        
        #Set the batch to start if all the training is complete, that is 1 epoch is complete.
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
        return x, y
    
#----------------------------------------------
        
#get logits:
model = GPT(GPTConfig())
model.to(device)
#logits, loss = model(x, y)

train_loader = DataLoaderLite(B=4, T=32)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4) # learning_rate(lr=3e-4) Adam optimizer is different than SGD(stochastic gradient decent method, But does the same work, adam is improved version SGD)
#Optmizing
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f'Step {i+1}, Loss : {loss.item()}')   # .item() --> converts tensor value to single float value.
    
'''This optimization is memorising the tokens, as loss function is approching zero. So we dont need it to 
memorise things. So we give it different data set, by data loader class up next.'''

print(logits.shape)
print(loss)

import sys
sys.exit(0)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim= -1)
        
        topk_probs, topk_indices = torch.topk(probs, 50, dim= -1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim= -1)

        '''x is of size (B, T), where B = 5, T = 8. setting seed = 42 to get the constant output.
        we are using torch.no_grad(), so that it doesnt have to store all the tensors in the 
        intermidiate step. which saves memory and a bit of time. 
        we are using hugging face top_k function, which samples the top 50 probabilities and clamps other probalities
        to zero and re-normalize the probabilities, which keep the model on track and makes sure
        model doesnt randomly go on generating the tokens.'''

#Print the generated text..
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded) 
    
print("It worked.......")