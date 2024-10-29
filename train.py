import torch
import random
from fft import BigramLanguageModel
import pickle
import re

torch.manual_seed(1337)
scaler = torch.amp.GradScaler('cuda')

batch_size = 64
time_intervals = 384
max_iter = 1000000
eval_interval = 250
learning_rate = 3e-5
eval_iters = 10



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


with open('./input/tokens.pkl', 'rb') as f:
    tokens  = pickle.load(f)

with open('./input/input.pkl', 'rb') as f:
    input_tokens  = pickle.load(f)

vocab_size = len(tokens)


print(tokens)
print('vocab token size: ', vocab_size)
print('input text token size: ', len(input_tokens))


stoi = {ch:i for i,ch in enumerate(tokens)}
itos = {i:ch for i,ch in enumerate(tokens)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])




data = torch.tensor(encode(input_tokens), dtype=torch.long)


torch.manual_seed(135665)


def get_batch():
    #var_time = time_intervals - 2*random.randint(0, 2)
    var_time = time_intervals
    ix = torch.randint(len(data) - var_time, (batch_size, ))
    x = torch.stack([data[i:i+var_time] for i in ix])
    y = torch.stack([data[i+1:i+var_time+1] for i in ix])
    return x.to(device), y.to(device)


def get_random_block():
    #var_time = time_intervals - 2*random.randint(0, 50)
    var_time = time_intervals
    i = random.randint(0, len(data) - var_time)
    block = data[i:i+var_time].reshape(1, -1).to(device)
    return block


@torch.no_grad()
def estimate_loss():
    LLM.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch()
        _, _, loss = LLM.update(X, targets=Y)
        losses[k] = loss.item()
    out = losses.mean()
    LLM.train()
    return out

def text_correct(text, multiline=True):
    def cap(match):
        return(match.group().capitalize())
    p = re.compile(r'(?<=[\.\?!][\s\n])(\w+)')
    text = text.replace(" i ", " I ")
    text = text.replace(" lord ", " Lord ")
    text = text.replace(" god ", " God ")
    for _ in range(7):
        text = text.replace(" : ", ": ")
        text = text.replace(" ! ", "! ")
        text = text.replace(" ? ", "? ")
        text = text.replace(" . ", ". ")
        text = text.replace(" , ", ", ")
    if multiline: text = text.replace(" * ", "\n")
    for _ in range(7): text = text.replace("  ", " ")


    text = p.sub(cap, text)
    return text


LLM = BigramLanguageModel(vocab_size, time_intervals, vocab_embed=738, n_embed=738, features=3, n_layers=17, device=device).to(device)
optimizer = torch.optim.AdamW(LLM.parameters(), lr=learning_rate)

pytorch_total_params = sum(p.numel() for p in LLM.parameters())
print('LLM parameters: ', pytorch_total_params)


try:
    LLM.load_state_dict(torch.load('LLM_model.pt',  weights_only=True))
    context = get_random_block()
    text = decode(LLM.generate(context, max_new_tokens=1000)[0].tolist())[-1000:]
    print(text_correct(text))

    print("loaded")
except:
    print("no LLM")

for iter in range(max_iter):


    if iter % eval_interval == 0:
        losses = estimate_loss()
        context = get_random_block()
        text = decode(LLM.generate(context, max_new_tokens=100)[0].tolist())[-100:]
        text = text_correct(text, multiline=False)

        print(f"step {iter}, train loss: {losses:.4f}, text: {text}")
        if iter>=500:
            try:
                torch.save(LLM.state_dict(), 'LLM_model.pt')
            except:
                print("problem during saving LLM")

        if iter>=10000 and iter%10000==0:
            context = get_random_block()
            text = decode(context[0].tolist())
            print(text_correct(text))
            print("###########################################")
            print("###########################################")
            text = decode(LLM.generate(context, max_new_tokens=500)[0].tolist())
            print(text_correct(text))
            print("###########################################")
            print("###########################################")

    #sample batch of data
    xb, yb = get_batch()

    #evaluate the loss
    with torch.amp.autocast('cuda', dtype=torch.float16):
        _, _, loss = LLM.update(xb, targets=yb)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


#generate from the LLM
#context = torch.ones((1,1), dtype=torch.long, device=device)

context = get_random_block()

text = decode(context[0].tolist())
print(text_correct(text))

print("###########################################")
print("###########################################")
print("###########################################")


text = decode(LLM.generate(context, max_new_tokens=500)[0].tolist())
print(text_correct(text))




