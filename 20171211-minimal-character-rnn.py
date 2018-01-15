# ==========================================================================
# Play with Recurrent Neural Net
# --------------------------------------------------------------------------
# Code Reference: Andrej Karpathy and Yoav Goldberg (links in the code)
#
# Practice RNN with only NumPy
# 
# Imagine neural unit as a function Fn(input_data) -> output_data. In a
#   typical feed forward neural nets, the output of a function is fed to
#   another neural unit. The final output becomes Fn3(Fn2(Fn1(input_data)))
#
# In RNN, the output of a function can be fed back to itself and becomes
#   Fn1(Fn1(Fn1(input_data))). Reusage of the same function seems reasonable
#   when the tasks handled by the neural units are the same. e.g. Reading 
#   the first sentence of a paragraph should not be that different from
#   reading the middle sentence. Reusing the same unit would also allow
#   faster back-propagation. 
#  
# Specifically, an RNN for character-level text generation would take in a
#   context (hidden state, "h") and a current character, to produce:
#       i) one output = the generated next character 
#       ii) one output = ("hidden state") = store the context so far
#  
# ==========================================================================

import numpy as np 

def RNN():
    '''
    Minimal character-level Vanilla RNN model.
    From Andrej Karpathy
    http://karpathy.github.io/2015/05/21/rnn-effectiveness/
    https://gist.github.com/karpathy/d4dee566867f8291f086
    '''

    def LossFun(inputs, target, hprev):
        '''
        inputs, targets are both list of integers
        hprev is Hx1 array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state
        '''
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        for t in range(len(inputs)):
            xs[t] = np.zeros((vocab_size, 1)) # [v,1]
            xs[t][inputs[t]] = 1 # Simple one hot encoding
            hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # [h,1] h(t) = tanh(wX + wh(t-1) + bh)
            ys[t] = np.dot(Why, hs[t]) + by # y = wh + by = [y, 1]
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            loss += -np.log(ps[t][targets[t], 0]) # Realize that the tensor is 3D

        dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
        dbh, dby = np.zeros_like(bh), np.zeros_like(by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t]) # [y,1]
            dy[targets[t]] -= 1 # Derive http://cs231n.github.io/neural-networks-case-study/#grad
            dWhy += np.dot(dy, hs[t].T) # [y,h]
            dby += dy # [y,1]
            dh = np.dot(Why.T, dy) + dhnext #+ dhnext, propagate
            dhraw = (1-hs[t]*hs[t])*dh 
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(Whh.T, dhraw)

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) # Clip to mitigate exploding gradients
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

    def sample(h, seed_ix, n):
        '''
        sample a sequence of integers from the model
        h is memory state, seed_ix is seed letter for first time step
        '''
        x = np.zeros((vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(n):
            h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
            y = np.dot(Why, h) + by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(vocab_size), p=p.ravel())
            x = np.zeros((vocab_size, 1))
            x[ix]=1
            ixes.append(ix)
        return ixes

    # data I/O
    data = open('tmp/index.txt', 'r', encoding='utf-8').read()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print('data has {} chars, {} unique'.format(data_size, vocab_size))
    char_to_ix = {ch:i for i,ch in enumerate(chars)}
    ix_to_char = {i:ch for i,ch in enumerate(chars)}

    # Hyperparam
    hidden_size = 100
    seq_length = 25
    learning_rate = 1e-1

    # Model Param
    Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
    Whh = np.random.randn(hidden_size, hidden_size) * 0.01
    Why = np.random.randn(vocab_size, hidden_size) * 0.01
    bh = np.zeros((hidden_size, 1)) 
    by = np.zeros((vocab_size, 1))

    n, p = 0, 0
    mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    mbh, mby = np.zeros_like(bh), np.zeros_like(by)
    smooth_loss = -np.log(1.0/vocab_size) * seq_length
    while True:
        if p+seq_length+1 >= len(data) or n==0:
            hprev = np.zeros((hidden_size,1)) # reset RNN memory
            p = 0
        inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
        targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

        if n%100 == 0:
            sample_ix = sample(hprev, np.random.randint(0, vocab_size), 100)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print("----\n {} \n----".format(txt))

        loss, dWxh, dWhh, dWhy, dbh, dby, hprev = LossFun(inputs, targets, hprev)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if n % 100 == 0:
            print("iter {}, loss {}".format(n, smooth_loss))

        for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                      [dWxh, dWhh, dWhy, dbh, dby],
                                      [mWxh, mWhh, mWhy, mbh, mby]):
            mem += dparam * dparam #? what is m
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

        p += seq_length
        n += 1

# From http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139
# Maximum Likelihood n-character predictor

def char_lm():

    def train_char_lm(fname, order=4):
        data = open(fname, 'r',encoding='utf-8').read()
        from collections import defaultdict, Counter
        lm = defaultdict(Counter)
        data += " " * order
        for i in range(len(data) - order):
            history, char = data[i:i+order], data[i+order]
            lm[history][char] += 1
        def normalize(counter):
            s = float(sum(counter.values()))
            return [(c,cnt/s) for c,cnt in counter.items()]
        return {hist:normalize(chars) for hist,chars in lm.items()}

    def genreate_letter(lm, history, candidates):
        if history not in lm: return np.random.choice(candidates)
        pred, prob = zip(*lm[history])
        return np.random.choice(pred, p=prob)

    def generate_text(lm, order, nletters=1000):
        history = " " * order
        candidates = [a[0] for hist in lm.values() for a in hist]
        candidates = list(set(candidates))
        out = []
        for i in range(nletters):
            c = genreate_letter(lm, history, candidates)
            history = history[1:] + c 
            out.append(c)
        return "".join(out)

    lm = train_char_lm("tmp/index.txt", order=2)
    print(generate_text(lm, 2))

RNN()
