# ----------------------------------------------------------
# StreamLit.py
# ----------------------------------------------------------

import streamlit as st
import numpy as np
import json

# Se importa DOAR din lstm_core.py (fara efecte secundare, fara antrenare)
from lstm_core import LSTM, construieste_arbore_huffman, obtine_coduri


@st.cache_resource
def load_trained_model():
    # 1. Se incarca vocabularul complet direct din JSON
    with open("vocab.json", "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    char2idx = vocab_data['char2idx']
    idx2char = {int(k): v for k, v in vocab_data['idx2char'].items()}
    vocab_size = len(char2idx)  # Dimensiunea reala pe care a fost antrenat modelul

    # 2. Incarcam ponderile din NPZ
    weights = np.load("lstm_weights.npz")
    hidden_size = int(weights['hidden_size'][0])
    seq_len = int(weights['seq_len'][0])

    # 3. Se initializeaza obiectul LSTM cu dimensiunea EXACTA din salvare
    loaded_lstm = LSTM(input_size=vocab_size, hidden_size=hidden_size, output_size=vocab_size)

    # 4. Injectam ponderile salvate
    loaded_lstm.Wf = weights['Wf']; loaded_lstm.bf = weights['bf']
    loaded_lstm.Wi = weights['Wi']; loaded_lstm.bi = weights['bi']
    loaded_lstm.Wg = weights['Wg']; loaded_lstm.bg = weights['bg']
    loaded_lstm.Wo = weights['Wo']; loaded_lstm.bo = weights['bo']
    loaded_lstm.Wy = weights['Wy']; loaded_lstm.by = weights['by']

    return loaded_lstm, char2idx, idx2char, vocab_size, seq_len


# Se apeleaza functia - toate variabilele sunt cele din salvare
lstm_model, char2idx, idx2char, vocab_size, seq_len = load_trained_model()


# --- CONFIGURARE INTERFATA GRAFICA ---
st.set_page_config(page_title="Smart Keyboard LSTM", layout="centered")

st.title("Simulator Smart Keyboard (Limba Română)")
st.write("Tastați în caseta de mai jos. Modelul LSTM va prezice live top 3 cele mai probabile litere care urmează.")

# Se initializeaza starea sesiunii pentru a retine textul introdus
if "text_introdus" not in st.session_state:
    st.session_state.text_introdus = "ana invata algo"


# Functie callback care se executa instant cand utilizatorul tasteaza
def la_modificare_text():
    st.session_state.text_introdus = st.session_state.tastatura_virtuala


# Se sincronizeaza cheia widget-ului cu textul INAINTE de render
# (inclusiv dupa ce un buton a adaugat o litera)
st.session_state.tastatura_virtuala = st.session_state.text_introdus

text_curent = st.text_input(
    "Tastați aici:",
    key="tastatura_virtuala",
    on_change=la_modificare_text
)

# Se sincronizeaza variabila folosita mai jos in cod
st.session_state.text_introdus = text_curent


# --- LOGICA DE PREDICTIE CARACTER URMATOR ---
if text_curent:
    # Se pregateste contextul (ultimele seq_len caractere)
    start = max(0, len(text_curent) - seq_len)
    context = text_curent[start:]
    while len(context) < seq_len:
        context = '0' + context

    # Se converteste contextul in vectori one-hot
    context_encoded = []
    for ch in context:
        # Daca utilizatorul scrie un caracter care nu e in vocabular, fallback pe spatiu
        ch_to_use = ch if ch in char2idx else ' '
        v = [0.0] * vocab_size
        v[char2idx[ch_to_use]] = 1.0
        context_encoded.append(np.array(v).reshape(-1, 1))

    # Se obtin probabilitatile din LSTM
    probs, _, _ = lstm_model.forward(context_encoded)
    probs = probs.flatten()

    # Se construieste arborele Huffman
    root = construieste_arbore_huffman(probs, idx2char)
    coduri = {}
    obtine_coduri(root, "", coduri)

    # Se extrag top 3 cele mai mari probabilitati
    top_3_indici = np.argsort(probs)[-3:][::-1]

    st.write("### Sugestii pentru următoarea literă:")

    # Se creeaza cele 3 butoane de sugestie
    col1, col2, col3 = st.columns(3)
    coloane = [col1, col2, col3]

    for i, idx in enumerate(top_3_indici):
        litera_sugerata = idx2char[idx]
        probabilitate = probs[idx] * 100

        # Se schimba vizual afisarea daca e vorba de un spatiu
        eticheta_buton = (
            f"'{litera_sugerata}' ({probabilitate:.1f}%)"
            if litera_sugerata != ' '
            else f"'[Spatiu]' ({probabilitate:.1f}%)"
        )

        # Daca se apasa pe butonul de sugestie, adaugam litera la text
        with coloane[i]:
            if st.button(eticheta_buton, key=f"btn_{i}", use_container_width=True):
                st.session_state.text_introdus += litera_sugerata
                st.rerun()