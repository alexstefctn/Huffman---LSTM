import numpy as np
import heapq


# ---------------------------------------------
# CLASA LSTM
# ---------------------------------------------

class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        d = input_size + hidden_size

        scale = 0.01
        self.Wf = np.random.randn(hidden_size, d) * scale
        self.Wi = np.random.randn(hidden_size, d) * scale
        self.Wg = np.random.randn(hidden_size, d) * scale
        self.Wo = np.random.randn(hidden_size, d) * scale

        self.bf = np.ones((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bg = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

        self.Wy = np.random.randn(output_size, hidden_size) * scale
        self.by = np.zeros((output_size, 1))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def step(self, x, h_prev, c_prev):
        xh = np.vstack([h_prev, x])
        f = self.sigmoid(self.Wf @ xh + self.bf)
        i = self.sigmoid(self.Wi @ xh + self.bi)
        g = np.tanh(self.Wg @ xh + self.bg)
        o = self.sigmoid(self.Wo @ xh + self.bo)
        c = f * c_prev + i * g
        h = o * np.tanh(c)
        return h, c, f, i, g, o

    def forward(self, seq):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        for x in seq:
            h, c, *_ = self.step(x, h, c)
        logits = self.Wy @ h + self.by
        probs = self.softmax(logits)
        return probs, h, c

    def train_step(self, seq, target_idx, lr=0.01):
        T = len(seq)
        hs = [np.zeros((self.hidden_size, 1))]
        cs = [np.zeros((self.hidden_size, 1))]
        fs, is_, gs, os = [], [], [], []

        for t in range(T):
            x = seq[t]
            h, c, f, i, g, o = self.step(x, hs[-1], cs[-1])
            hs.append(h); cs.append(c)
            fs.append(f); is_.append(i); gs.append(g); os.append(o)

        logits = self.Wy @ hs[-1] + self.by
        probs = self.softmax(logits)
        loss = -np.log(probs[target_idx, 0] + 1e-9)

        dlogits = probs.copy()
        dlogits[target_idx] -= 1.0

        dWy = dlogits @ hs[-1].T
        dby = dlogits
        dh = self.Wy.T @ dlogits

        dWf = np.zeros_like(self.Wf); dWi = np.zeros_like(self.Wi)
        dWg = np.zeros_like(self.Wg); dWo = np.zeros_like(self.Wo)
        dbf = np.zeros_like(self.bf); dbi = np.zeros_like(self.bi)
        dbg = np.zeros_like(self.bg); dbo = np.zeros_like(self.bo)

        dc = np.zeros((self.hidden_size, 1))

        for t in reversed(range(T)):
            x = seq[t]
            do = dh * np.tanh(cs[t + 1])
            dc += dh * os[t] * (1.0 - np.tanh(cs[t + 1]) ** 2)
            df = dc * cs[t]
            di = dc * gs[t]
            dg = dc * is_[t]
            dc_prev = dc * fs[t]

            d_gate_f = df * fs[t] * (1 - fs[t])
            d_gate_i = di * is_[t] * (1 - is_[t])
            d_gate_g = dg * (1 - gs[t] ** 2)
            d_gate_o = do * os[t] * (1 - os[t])

            xh = np.vstack([hs[t], x])

            dWf += d_gate_f @ xh.T; dbf += d_gate_f
            dWi += d_gate_i @ xh.T; dbi += d_gate_i
            dWg += d_gate_g @ xh.T; dbg += d_gate_g
            dWo += d_gate_o @ xh.T; dbo += d_gate_o

            dxh = (self.Wf.T @ d_gate_f + self.Wi.T @ d_gate_i +
                   self.Wg.T @ d_gate_g + self.Wo.T @ d_gate_o)
            dh = dxh[:self.hidden_size]
            dc = dc_prev

        for grad in [dWf, dWi, dWg, dWo, dbf, dbi, dbg, dbo, dWy, dby]:
            np.clip(grad, -5, 5, out=grad)

        self.Wf -= lr * dWf; self.bf -= lr * dbf
        self.Wi -= lr * dWi; self.bi -= lr * dbi
        self.Wg -= lr * dWg; self.bg -= lr * dbg
        self.Wo -= lr * dWo; self.bo -= lr * dbo
        self.Wy -= lr * dWy; self.by -= lr * dby

        return loss


# ---------------------------------------------
# ARBORE HUFFMAN
# ---------------------------------------------

class HuffmanNode:
    def __init__(self, char, prob):
        self.char = char
        self.prob = prob
        self.left = None
        self.right = None


def construieste_arbore_huffman(probs, idx2char):
    """
    Construieste arborele Huffman pe baza vectorului de probabilitati.
    idx2char: dict {int -> char}, primit explicit (nu variabila globala).
    """
    heap = []
    for i, p in enumerate(probs):
        char = idx2char[i]
        node = HuffmanNode(char, float(p))
        heapq.heappush(heap, (float(p), i, node))

    tie_breaker = len(probs)
    while len(heap) > 1:
        p1, _, left = heapq.heappop(heap)
        p2, _, right = heapq.heappop(heap)

        merged_prob = p1 + p2
        merged = HuffmanNode(None, merged_prob)
        merged.left = left
        merged.right = right

        heapq.heappush(heap, (merged_prob, tie_breaker, merged))
        tie_breaker += 1

    _, _, root = heapq.heappop(heap)
    return root


def obtine_coduri(nod, cod_curent, dictionar):
    if nod is None:
        return
    if nod.char is not None:
        dictionar[nod.char] = cod_curent if cod_curent else "0"
        return
    obtine_coduri(nod.left, cod_curent + "0", dictionar)
    obtine_coduri(nod.right, cod_curent + "1", dictionar)
