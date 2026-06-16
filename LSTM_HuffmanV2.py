"""
LSTM + Adaptive Huffman Coding
"""

import numpy as np
import heapq
from collections import Counter

def entropie_empirica(text):
    frecvente = Counter(text)
    n = len(text)
    return -sum((f/n) * np.log2(f/n) for f in frecvente.values())

# ---------------------------------------------
# 0. CORPUS
# ---------------------------------------------

CORPUS = ''

#CORPUS = "ana are mere si pere. ana invata algoritmi de compresie neuronala. ana are mere si pere. ana invata algoritmi de compresie neuronala.ana are mere si pere. ana invata algoritmi de compresie neuronala.ana are mere si pere. ana invata algoritmi de compresie neuronala.ana are mere si pere. ana invata algoritmi de compresie neuronala."
CORPUS_TEST = CORPUS

if CORPUS == '':
    file = open("corpus_romana.txt", "r", encoding="utf-8")
    CORPUS_INTREG = file.read().strip().lower()
    file.close()
    split = int(len(CORPUS_INTREG) * 0.8)
    CORPUS = CORPUS_INTREG[:split]   # 80% - pentru antrenare
    CORPUS_TEST  = CORPUS_INTREG[split:]   # 20% - nevazut de model


# ---------------------------------------------
# 1. VOCABULAR
# ---------------------------------------------

# IMPORTANT: Se forteaza includerea caracterului '0' in vocabular pentru padding
chars = sorted(set(CORPUS + '0'))  
vocab_size = len(chars)

char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}

print(f"Vocabular: {vocab_size} caractere unice (inclusiv padding '0')")
print(f"Caractere: {repr(''.join(chars))}")


# ---------------------------------------------
# 2. DATE DE ANTRENARE (MODIFICAT)
# ---------------------------------------------

SEQ_LEN = 10

X_data = []
y_data = []

# Se porneste de la indexul 0 pentru ca modelul sa invete sa prezica pe baza de '0' (padding)
for i in range(len(CORPUS)):
    target = CORPUS[i]
    
    # Se construieste contextul (istoricul)
    start = max(0, i - SEQ_LEN)
    context = CORPUS[start:i]
    
    # Daca este la inceput, se umple cu '0'
    while len(context) < SEQ_LEN:
        context = '0' + context
        
    # Se codifica tot contextul in one-hot
    seq_encoded = []
    for ch in context:
        one_hot = [0.0] * vocab_size
        one_hot[char2idx[ch]] = 1.0
        seq_encoded.append(one_hot)

    X_data.append(seq_encoded)
    y_data.append(char2idx[target])

X = np.array(X_data)
y = np.array(y_data)

print(f"Date de antrenare: {X.shape[0]} exemple")

# ---------------------------------------------
# 3. CLASA LSTM
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

        dWf=np.zeros_like(self.Wf); dWi=np.zeros_like(self.Wi)
        dWg=np.zeros_like(self.Wg); dWo=np.zeros_like(self.Wo)
        dbf=np.zeros_like(self.bf); dbi=np.zeros_like(self.bi)
        dbg=np.zeros_like(self.bg); dbo=np.zeros_like(self.bo)

        dc = np.zeros((self.hidden_size, 1))

        for t in reversed(range(T)):
            x = seq[t]
            do = dh * np.tanh(cs[t+1])
            dc += dh * os[t] * (1.0 - np.tanh(cs[t+1])**2)
            df = dc * cs[t]
            di = dc * gs[t]
            dg = dc * is_[t]
            dc_prev = dc * fs[t]

            d_gate_f = df * fs[t] * (1 - fs[t])
            d_gate_i = di * is_[t] * (1 - is_[t])
            d_gate_g = dg * (1 - gs[t]**2)
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

        for grad in [dWf,dWi,dWg,dWo,dbf,dbi,dbg,dbo,dWy,dby]:
            np.clip(grad, -5, 5, out=grad)

        self.Wf -= lr * dWf; self.bf -= lr * dbf
        self.Wi -= lr * dWi; self.bi -= lr * dbi
        self.Wg -= lr * dWg; self.bg -= lr * dbg
        self.Wo -= lr * dWo; self.bo -= lr * dbo
        self.Wy -= lr * dWy; self.by -= lr * dby

        return loss

# ---------------------------------------------
# 4. ANTRENARE
# ---------------------------------------------

HIDDEN_SIZE = 64
EPOCI       = 100
LR          = 0.02  #0.005

lstm = LSTM(input_size=vocab_size, hidden_size=HIDDEN_SIZE, output_size=vocab_size)

history_loss = []

print("\n- Antrenare LSTM -")
for EPOCA in range(EPOCI):
    total_loss = 0.0
    indices = np.random.permutation(len(X_data))

    for idx in indices:
        seq = [np.array(X_data[idx][t]).reshape(-1, 1) for t in range(SEQ_LEN)]
        loss = lstm.train_step(seq, y_data[idx], lr=LR)
        total_loss += loss

    avg_loss = total_loss / len(X_data)
    history_loss.append(avg_loss)          # Graficul
    if (EPOCA + 1) % 5 == 0:
        print(f"  EPOCA {EPOCA+1:2d}/{EPOCI}  |  Loss mediu: {avg_loss:.4f}")

# ---------------------------------------------
# 4b. GRAFIC EVOLUTIE LOSS
# ---------------------------------------------

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(range(1, EPOCI + 1), history_loss,
        color="#4a90d9", linewidth=1.8, label="Loss mediu per epoca")

ax.set_xlabel("Epoca")
ax.set_ylabel("Loss mediu")
ax.set_title("Evolutia pierderii in antrenare (LSTM)")
ax.legend()
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150, bbox_inches='tight')
plt.show()

print("Graficul loss-ului a fost salvat ca 'loss_curve.png'")

# ---------------------------------------------
# 5. ARBORELE HUFFMAN ADAPTIV
# ---------------------------------------------

class HuffmanNode:
    def __init__(self, char, prob):
        self.char = char
        self.prob = prob
        self.left = None
        self.right = None

def construieste_arbore_huffman(probs):
    heap = []
    # Folosim index-ul i pentru a garanta determinismul cand probabilitatile sunt egale
    for i, p in enumerate(probs):
        char = idx2char[i]
        node = HuffmanNode(char, float(p))
        # Tuplul din heap: (probabilitate, tie_breaker, nod)
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
    if nod is None: return
    if nod.char is not None:
        dictionar[nod.char] = cod_curent if cod_curent else "0"
        return
    obtine_coduri(nod.left, cod_curent + "0", dictionar)
    obtine_coduri(nod.right, cod_curent + "1", dictionar)

# ---------------------------------------------
# 6. FUNCTII DE COMPRIMARE / DECOMPRIMARE
# ---------------------------------------------

def comprima(text, lstm):
    encoded_bits = ""
    for i, caracter in enumerate(text):
        start = max(0, i - SEQ_LEN)
        context = text[start:i]
        while len(context) < SEQ_LEN:
            context = '0' + context
            
        context_encoded = []
        for ch in context:
            v = [0.0] * vocab_size
            v[char2idx[ch]] = 1.0
            context_encoded.append(np.array(v).reshape(-1, 1))
            
        probs, _, _ = lstm.forward(context_encoded)
        probs = probs.flatten()
        
        root = construieste_arbore_huffman(probs)
        dictionar = {}
        obtine_coduri(root, "", dictionar)
        
        encoded_bits += dictionar[caracter]
        
    return encoded_bits

def decompress(encoded_bits, lstm, lungime_text):
    decoded_text = ""
    bit_index = 0
    
    while len(decoded_text) < lungime_text:
        start = max(0, len(decoded_text) - SEQ_LEN)
        context = decoded_text[start:]
        while len(context) < SEQ_LEN:
            context = '0' + context
            
        context_encoded = []
        for ch in context:
            v = [0.0] * vocab_size
            v[char2idx[ch]] = 1.0
            context_encoded.append(np.array(v).reshape(-1, 1))
            
        probs, _, _ = lstm.forward(context_encoded)
        probs = probs.flatten()
        
        root = construieste_arbore_huffman(probs)
        
        current_node = root
        while current_node.char is None:
            if bit_index >= len(encoded_bits):
                break
            bit = encoded_bits[bit_index]
            bit_index += 1
            if bit == "0":
                current_node = current_node.left
            else:
                current_node = current_node.right
                
        decoded_text += current_node.char
        
    return decoded_text

# ---------------------------------------------
# 7. TESTARE
# ---------------------------------------------

test_text = "ana are mere si pere. ana invata algoritmi de compresie neuronala."
#test_text = CORPUS_TEST[:200]  # primele 200 caractere nevazute

print(f"\n- Test Compresie & Decompresie -")
print(f"Text original: '{test_text}'")

biti_comprimati = comprima(test_text, lstm)
print(f"Codificat:     {biti_comprimati}")

# La decompresie trebuie sa stim cate litere cautam
text_reconstruit = decompress(biti_comprimati, lstm, len(test_text))
print(f"Decodificat:   '{text_reconstruit}'")

biti_ascii = len(test_text) * 8
biti_huffman = len(biti_comprimati)
print(f"\nStatistici:")
print(f"Biti originali (ASCII): {biti_ascii}")
print(f"Biti dupa model LSTM:   {biti_huffman}")
print(f"Rata compresie:         {100*(1 - biti_huffman/biti_ascii):.1f}%")


def autocomplete(prompt, lstm, max_steps=20):
    text_generat = prompt
    print(f"\nPrompt introdus: '{prompt}'")
    print("Modelul completeaza cuvantul: ", end="")
    
    for _ in range(max_steps):
        # 1. Se construieste contextul din ultimele caractere ale textului curent
        start = max(0, len(text_generat) - SEQ_LEN)
        context = text_generat[start:]
        while len(context) < SEQ_LEN:
            context = '0' + context
            
        # 2. Se converteste contextul in vectori one-hot
        context_encoded = []
        for ch in context:
            v = [0.0] * vocab_size
            v[char2idx[ch]] = 1.0
            context_encoded.append(np.array(v).reshape(-1, 1))
            
        # 3. Se ruleaza LSTM-ul pentru a se observa ce litera urmeaza
        probs, _, _ = lstm.forward(context_encoded)
        probs = probs.flatten()
        
        # 4. Se alege caracterul cu probabilitatea cea mai mare (Argmax)
        urmatorul_idx = np.argmax(probs)
        litera_prezisa = idx2char[urmatorul_idx]
        
        # 5. Conditia de oprire: daca a prezis spatiu sau punct, se opreste ca sa nu halucineze mai departe
        if litera_prezisa == ' ' or litera_prezisa == '.':
            break
            
        print(litera_prezisa, end="", flush=True)
        text_generat += litera_prezisa
        
    print("\n[Oprire generare: S-a detectat sfarsitul cuvantului]")
    return text_generat

# --- TESTARE AUTOCOMPLETE ---

cuvant_incomplet = "ana invata algorit" 
text_final = autocomplete(cuvant_incomplet, lstm)
print(f"Rezultat final autocomplete: '{text_final}'")

# ---------------------------------------------
# SALVARE PONDERI ȘI VOCABULAR
# ---------------------------------------------

print("\n- Salvare model antrenat... -")

# Cream un dictionar cu toate greutatile din obiectul lstm
model_weights = {
    'Wf': lstm.Wf, 'Wi': lstm.Wi, 'Wg': lstm.Wg, 'Wo': lstm.Wo,
    'bf': lstm.bf, 'bi': lstm.bi, 'bg': lstm.bg, 'bo': lstm.bo,
    'Wy': lstm.Wy, 'by': lstm.by,
    # Salvam si hiperparametrii retelei
    'hidden_size': np.array([HIDDEN_SIZE]),
    'seq_len': np.array([SEQ_LEN])
}

# Se salveaza greutatile intr-fișier numit 'lstm_weights.npz'
np.savez("lstm_weights.npz", **model_weights)

# De asemenea, trebuie salvat vocabularul, altfel decodorul nu va sti 
# ce caracter corespunde fiecarui index
import json
vocab_data = {
    'chars': chars,
    'char2idx': char2idx,
    # In JSON cheile trebuie sa fie stringuri, asa ca se va converti idx2char la string-cheie
    'idx2char': {str(k): v for k, v in idx2char.items()}
}

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab_data, f, ensure_ascii=False, indent=4)

print("Modelul 'lstm_weights.npz' si vocabularul 'vocab.json' au fost salvate cu succes!")

# ---------------------------------------------
# 8. COMPARATIE: HUFFMAN STATIC vs HUFFMAN + LSTM
# ---------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

# --- 8.1 Huffman Static ---

def construieste_arbore_huffman_static(frecvente):
    heap = []
    for i, (char, freq) in enumerate(frecvente.items()):
        node = HuffmanNode(char, freq)
        heapq.heappush(heap, (freq, i, node))

    tie_breaker = len(frecvente)
    while len(heap) > 1:
        p1, _, left = heapq.heappop(heap)
        p2, _, right = heapq.heappop(heap)
        merged = HuffmanNode(None, p1 + p2)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, (p1 + p2, tie_breaker, merged))
        tie_breaker += 1

    _, _, root = heapq.heappop(heap)
    return root

def comprima_static(text):
    frecvente = Counter(text)
    root = construieste_arbore_huffman_static(frecvente)
    coduri = {}
    obtine_coduri(root, "", coduri)
    return "".join(coduri[c] for c in text), coduri

# --- 8.2 Se ruleaza ambele pe test_text ---

bits_static, coduri_static = comprima_static(test_text)
bits_lstm = biti_comprimati  # deja calculat mai sus in sectiunea 7

biti_ascii   = len(test_text) * 8
biti_static  = len(bits_static)
biti_lstm_n  = len(bits_lstm)

print(f"\n- Comparatie Huffman Static vs LSTM -")
print(f"ASCII (referinta):   {biti_ascii} biti")
print(f"Huffman static:      {biti_static} biti  ({100*(1-biti_static/biti_ascii):.1f}% compresie)")
print(f"Huffman + LSTM:      {biti_lstm_n} biti  ({100*(1-biti_lstm_n/biti_ascii):.1f}% compresie)")

# --- 8.3 Lungimea codului per caracter (character-level) ---

lungimi_static = [len(coduri_static[c]) for c in test_text]

# Recalculam lungimile per caracter si pentru LSTM
lungimi_lstm = []
entropii_shannon = []

bit_idx = 0
for i, caracter in enumerate(test_text):
    start = max(0, i - SEQ_LEN)
    context = test_text[start:i]
    while len(context) < SEQ_LEN:
        context = '0' + context

    context_encoded = []
    for ch in context:
        v = [0.0] * vocab_size
        v[char2idx[ch]] = 1.0
        context_encoded.append(np.array(v).reshape(-1, 1))

    probs, _, _ = lstm.forward(context_encoded)
    probs_flat = probs.flatten()
    
    root_temp = construieste_arbore_huffman(probs_flat)
    coduri_temp = {}
    obtine_coduri(root_temp, "", coduri_temp)

    lungime_cod = len(coduri_temp[caracter])
    lungimi_lstm.append(lungime_cod)
    bit_idx += lungime_cod

    # Calculul entropiei Shannon locale H(X) pe distributia prezisa
    entropie_locala = -sum(p * np.log2(p) for p in probs_flat if p > 0)
    entropii_shannon.append(entropie_locala)

# --- AFISARE INTEGRALA A TEOREMEI LUI SHANNON ---
H_empiric = entropie_empirica(test_text)
lungime_medie_cod_lstm = np.mean(lungimi_lstm)
lungime_medie_cod_static = np.mean(lungimi_static)

entropii_lstm = []
for i, caracter in enumerate(test_text):
    start = max(0, i - SEQ_LEN)
    context = test_text[start:i]
    while len(context) < SEQ_LEN:
        context = '0' + context
    context_encoded = []
    for ch in context:
        v = [0.0] * vocab_size
        v[char2idx[ch]] = 1.0
        context_encoded.append(np.array(v).reshape(-1, 1))
    probs, _, _ = lstm.forward(context_encoded)
    probs_flat = probs.flatten()
    H_local = -sum(p * np.log2(p) for p in probs_flat if p > 0)
    entropii_lstm.append(H_local)

H_lstm = np.mean(entropii_lstm)

print(f"\n- Verificarea Teoremei lui Shannon -")
print(f"\nHuffman Static (distributie empirica):")
print(f"  H(X) empiric:        {H_empiric:.4f} biti/caracter")
print(f"  L_med static:        {lungime_medie_cod_static:.4f} biti/caracter")
print(f"  {H_empiric:.4f} <= {lungime_medie_cod_static:.4f} <= {H_empiric+1:.4f} : {'OK' if H_empiric <= lungime_medie_cod_static <= H_empiric+1 else 'EROARE'}")

print(f"\nHuffman + LSTM (distributie model):")
print(f"  H(X) model LSTM:     {H_lstm:.4f} biti/caracter")
print(f"  L_med LSTM:          {lungime_medie_cod_lstm:.4f} biti/caracter")
print(f"  {H_lstm:.4f} <= {lungime_medie_cod_lstm:.4f} <= {H_lstm+1:.4f} : {'OK' if H_lstm <= lungime_medie_cod_lstm <= H_lstm+1 else 'EROARE'}")

# --- 8.4 GRAFICE ---

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Comparatie Huffman Static vs Huffman + LSTM", fontsize=13, fontweight='bold')

# -- Grafic 1: Bar chart total biti --
ax1 = axes[0]
categorii  = ["ASCII\n(referinta)", "Huffman\nStatic", "Huffman\n+ LSTM"]
valori     = [biti_ascii, biti_static, biti_lstm_n]
culori     = ["#b0b0b0", "#4a90d9", "#e07b39"]

bars = ax1.bar(categorii, valori, color=culori, width=0.5, edgecolor="white")

for bar, val in zip(bars, valori):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        str(val),
        ha='center', va='bottom', fontsize=10, fontweight='bold'
    )

ax1.set_ylabel("Număr de biți")
ax1.set_title("Total biți pentru textul de test")
ax1.set_ylim(0, max(valori) * 1.15)
ax1.spines[['top', 'right']].set_visible(False)

# -- Grafic 2: Lungime cod per caracter --
ax2 = axes[1]
x = range(len(test_text))

ax2.plot(x, lungimi_static, color="#4a90d9", linewidth=1.2,
         linestyle="--", label="Huffman Static", alpha=0.85)
ax2.plot(x, lungimi_lstm,   color="#e07b39", linewidth=1.2,
         label="Huffman + LSTM", alpha=0.85)

ax2.set_xlabel("Poziția caracterului în text")
ax2.set_ylabel("Lungimea codului (biți)")
ax2.set_title("Lungimea codului per caracter")
ax2.legend()
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()

# (Figura 5.3.1)
extent_stang = axes[0].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
fig.savefig("figura_5_3_1_total_biti.png", bbox_inches=extent_stang.expanded(1.15, 1.2), dpi=150)
plt.show()

# (Figura 5.3.2)
extent_drept = axes[1].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
fig.savefig("figura_5_3_2_lungime_per_caracter.png", bbox_inches=extent_drept.expanded(1.15, 1.2), dpi=150)
plt.show()

# Figura completa
#plt.tight_layout()
#plt.savefig("comparatie_huffman.png", dpi=150, bbox_inches='tight')
#plt.show()

print("\nGraficul a fost salvat drept 'comparatie_huffman.png'")