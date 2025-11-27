import streamlit as st
import numpy as np
from PIL import Image

# ======================================================
#        BITWISE UTILITIES FOR ALL SECRET TYPES
# ======================================================
def text_to_bits(text):
    arr = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    bits = np.unpackbits(arr)
    return bits.reshape(1, -1)  # shape (1, N)

def bits_to_text(bits):
    arr = np.packbits(bits.reshape(-1))
    try:
        return arr.tobytes().decode("utf-8")
    except:
        return "<Decoding Error – corrupted text>"

def int_to_bits(n, bit_length=128):
    b = np.array(list(np.binary_repr(n, width=bit_length)), dtype=np.uint8)
    return b.reshape(1, -1)

def bits_to_int(bits):
    s = "".join(bits.reshape(-1).astype(str))
    return int(s, 2)

def load_image_bits(pil_image):
    img = pil_image.convert("L")
    arr = np.array(img)
    bits = np.unpackbits(arr, axis=1)
    return arr, bits

def bits_to_image(bits, shape):
    flat = np.packbits(bits.reshape(-1))
    return Image.fromarray(flat.reshape(shape).astype(np.uint8))


# ======================================================
#                  (2,4) SECRET SHARING
# ======================================================
def encode_24(bits):
    n = bits.shape[1] // 2
    M1 = bits[:, :n]
    M2 = bits[:, n:]

    R1 = np.random.randint(0,2,M1.shape,dtype=np.uint8)
    R2 = np.random.randint(0,2,M2.shape,dtype=np.uint8)

    E1 = np.concatenate([R1,     M2 ^ R2], axis=1)
    E2 = np.concatenate([M1 ^ R1, R2    ], axis=1)
    E3 = np.concatenate([M1 ^ R2, M2 ^ R1], axis=1)
    E4 = np.concatenate([M1 ^ M2 ^ R1, M1 ^ M2 ^ R2], axis=1)

    return [E1, E2, E3, E4]


# ===== DECODERS FOR ALL 2-SHARE CASES =====
def decode(Ea, Eb, a, b):
    n = Ea.shape[1] // 2
    if {a,b} == {1,2}:
        R1 = Ea[:, :n]
        X  = Ea[:, n:]
        Y  = Eb[:, :n]
        R2 = Eb[:, n:]
        M1 = Y ^ R1
        M2 = X ^ R2
        return np.concatenate([M1, M2], axis=1)

    elif {a,b} == {1,3}:
        R1 = Ea[:, :n]
        X  = Ea[:, n:]
        A  = Eb[:, :n]
        B  = Eb[:, n:]
        M2 = B ^ R1
        R2 = A ^ M2
        M1 = X ^ R2
        return np.concatenate([M1, M2], axis=1)

    elif {a,b} == {1,4}:
        R1 = Ea[:, :n]
        X  = Ea[:, n:]
        Y  = Eb[:, :n]
        Z  = Eb[:, n:]
        R2 = (Z ^ Y)  # derived expression
        M2 = X ^ R2
        M1 = Y ^ M2 ^ R1
        return np.concatenate([M1, M2], axis=1)

    elif {a,b} == {2,3}:
        R2 = Ea[:, n:]
        Y  = Ea[:, :n]
        A  = Eb[:, :n]
        B  = Eb[:, n:]
        R1 = B ^ (Y ^ A)
        M1 = Y ^ R1
        M2 = B ^ R1
        return np.concatenate([M1, M2], axis=1)

    elif {a,b} == {2,4}:
        Y = Ea[:, :n]
        R2 = Ea[:, n:]
        Z = Eb[:, n:]
        M2 = Z ^ R2
        M1 = Y ^ (Z ^ R2)
        return np.concatenate([M1, M2], axis=1)

    elif {a,b} == {3,4}:
        A = Ea[:, :n]
        B = Ea[:, n:]
        Y = Eb[:, :n]
        Z = Eb[:, n:]
        R1 = B ^ (Y ^ A)
        M2 = B ^ R1
        M1 = A ^ M2
        return np.concatenate([M1, M2], axis=1)

    else:
        return None


# ======================================================
#                   STREAMLIT UI
# ======================================================
st.title("Generalized (2,4) XOR Secret Sharing Demo")

mode = st.selectbox(
    "Choose secret type:",
    ["Image", "Text", "Integer"]
)

# ---------------------------------------------
# IMAGE MODE
# ---------------------------------------------
if mode == "Image":
    st.header("Image Secret Sharing")

    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if uploaded:
        pil = Image.open(uploaded)
        if max(pil.size) > 512:
            pil = pil.resize((512, 512))

        arr, bits = load_image_bits(pil)
        shares = encode_24(bits)

        st.subheader("Select shares for reconstruction")
        choices = st.multiselect("Pick shares:", ["E1","E2","E3","E4"])

        idx = [("E1",1),("E2",2),("E3",3),("E4",4)]
        selected_nums = [n for (label,n) in idx if label in choices]

        if len(selected_nums) == 0:
            st.info("No shares selected.")
        elif len(selected_nums) == 1:
            st.warning("A single share reveals **zero information**. Perfect secrecy.")
        else:
            # reconstruct from first two shares chosen
            a, b = selected_nums[:2]
            Ea = shares[a-1]
            Eb = shares[b-1]
            rec = decode(Ea, Eb, a, b)
            img = bits_to_image(rec, arr.shape)
            st.image(img, caption=f"Reconstruction from {choices[:2]}")


# ---------------------------------------------
# TEXT MODE
# ---------------------------------------------
elif mode == "Text":
    st.header("Text Secret Sharing")
    text = st.text_input("Enter secret text:")

    if text:
        bits = text_to_bits(text)
        shares = encode_24(bits)

        choices = st.multiselect("Pick shares:", ["E1","E2","E3","E4"])
        selected_nums = [
            {"E1":1,"E2":2,"E3":3,"E4":4}[c] for c in choices
        ]

        if len(selected_nums) == 0:
            st.info("No shares selected.")
        elif len(selected_nums) == 1:
            st.warning("A single share reveals **zero information**.")
        else:
            a, b = selected_nums[:2]
            rec = decode(shares[a-1], shares[b-1], a, b)
            st.success("Reconstruction:")
            st.write(bits_to_text(rec))


# ---------------------------------------------
# INTEGER MODE
# ---------------------------------------------
else:
    st.header("Integer Secret Sharing")

    int_str = st.text_input(
        "Enter integer (decimal):",
        value="123456",
        help="Any size integer is allowed. Will be encoded into 128 bits unless resized."
    )

    try:
        n = int(int_str)
    except:
        st.error("Please enter a valid integer.")
        st.stop()

    # convert to 128-bit fixed length binary
    bits = int_to_bits(n, bit_length=128)
    shares = encode_24(bits)

    choices = st.multiselect("Pick shares:", ["E1","E2","E3","E4"])
    selected_nums = [
        {"E1":1,"E2":2,"E3":3,"E4":4}[c] for c in choices
    ]

    if len(selected_nums) == 0:
        st.info("No shares selected.")
    elif len(selected_nums) == 1:
        st.warning("A single share reveals **zero information**.")
    else:
        a, b = selected_nums[:2]
        rec = decode(shares[a-1], shares[b-1], a, b)
        st.success("Reconstructed integer:")
        st.write(bits_to_int(rec))
