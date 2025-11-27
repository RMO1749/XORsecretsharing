import streamlit as st
import numpy as np
from PIL import Image

# ---------------------------
# Load image and convert to grayscale bits
# ---------------------------
def load_image_bits_pil(pil_image: Image.Image):
    img = pil_image.convert("L")  # grayscale only
    arr = np.array(img)
    bits = np.unpackbits(arr, axis=1)  # shape: (H, W*8)
    return arr, bits


# ---------------------------
# EXACT (2,4) ENCODING SCHEME FROM YOUR SLIDE
# ---------------------------
def encode_24(bits: np.ndarray):
    """
    bits: shape (H, W*8)
    Split horizontally into M1, M2 and apply your (2,4) XOR-based scheme.
    """
    n = bits.shape[1] // 2  # half the bit-length

    # Split message bits into halves M1, M2
    M1 = bits[:, :n]
    M2 = bits[:, n:]

    # Random masks
    R1 = np.random.randint(0, 2, M1.shape, dtype=np.uint8)
    R2 = np.random.randint(0, 2, M2.shape, dtype=np.uint8)

    # According to your slide:

    # E1 = (R1, M2 ⊕ R2)
    E1 = np.concatenate([R1, M2 ^ R2], axis=1)

    # E2 = (M1 ⊕ R1, R2)
    E2 = np.concatenate([M1 ^ R1, R2], axis=1)

    # E3 = (M1 ⊕ R2, M2 ⊕ R1)
    E3 = np.concatenate([M1 ^ R2, M2 ^ R1], axis=1)

    # E4 = (M1 ⊕ M2 ⊕ R1,  M1 ⊕ M2 ⊕ R2)
    E4 = np.concatenate([M1 ^ M2 ^ R1, M1 ^ M2 ^ R2], axis=1)

    return E1, E2, E3, E4


# ---------------------------
# DECODING — using (E1, E2)
# ---------------------------
def decode_from_E1_E2(E1: np.ndarray, E2: np.ndarray):
    """
    Decode using the algebra derived from:
      E1 = (R1, M2 ⊕ R2)
      E2 = (M1 ⊕ R1, R2)
    """
    n = E1.shape[1] // 2

    R1 = E1[:, :n]
    X  = E1[:, n:]      # M2 ⊕ R2

    Y  = E2[:, :n]      # M1 ⊕ R1
    R2 = E2[:, n:]

    M1 = Y ^ R1
    M2 = X ^ R2

    return np.concatenate([M1, M2], axis=1)


# ---------------------------
# Convert bits back to image cleanly
# ---------------------------
def bits_to_image(bits: np.ndarray, shape):
    """
    bits: shape (H, W*8) or equivalent
    shape: original image shape (H, W)
    """
    flat = np.packbits(bits.reshape(-1))
    return Image.fromarray(flat.reshape(shape).astype(np.uint8))


# ======================================================
#                  STREAMLIT APP
# ======================================================

st.title("2-of-4 XOR Secret Sharing Demo (Grayscale)")

st.markdown(
    """
Upload an image, and this app will:

1. Convert it to grayscale  
2. Encode it with the $(2,4)$ XOR-based secret-sharing scheme  
3. Show the four **shares** (each looks like random noise)  
4. Reconstruct the original image from shares $(E_1, E_2)$  
"""
)

uploaded_file = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image from upload
    pil_img = Image.open(uploaded_file)

    # Optional: resize large images to keep the demo snappy
    max_side = 512
    if max(pil_img.size) > max_side:
        pil_img = pil_img.resize(
            (int(pil_img.width * max_side / max(pil_img.size)),
             int(pil_img.height * max_side / max(pil_img.size))),
            Image.LANCZOS,
        )

    st.subheader("Original Image (grayscale)")
    st.image(pil_img.convert("L"), use_column_width=True)

    # Run encoding
    orig_arr, orig_bits = load_image_bits_pil(pil_img)
    E1, E2, E3, E4 = encode_24(orig_bits)

    # Reconstruct from (E1, E2)
    recovered_bits = decode_from_E1_E2(E1, E2)
    recovered_img = bits_to_image(recovered_bits, orig_arr.shape)

    st.subheader("Shares (E₁, E₂, E₃, E₄) — each looks like random noise")
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    c1.image(bits_to_image(E1, orig_arr.shape), caption="Share E₁", use_column_width=True)
    c2.image(bits_to_image(E2, orig_arr.shape), caption="Share E₂", use_column_width=True)
    c3.image(bits_to_image(E3, orig_arr.shape), caption="Share E₃", use_column_width=True)
    c4.image(bits_to_image(E4, orig_arr.shape), caption="Share E₄", use_column_width=True)

    st.subheader("Reconstruction from (E₁, E₂)")
    st.image(recovered_img, caption="Reconstructed Image from Shares E₁ and E₂", use_column_width=True)

    # Check perfect reconstruction
    equal = np.array_equal(np.array(recovered_img), orig_arr)
    st.markdown(
        f"**Perfect reconstruction:** {'✅ Yes' if equal else '❌ No (something is off)'}"
    )
else:
    st.info("⬆️ Upload an image to begin the demo.")
