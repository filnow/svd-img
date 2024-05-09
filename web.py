import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


def compression_rate(it: int, Ms: int, Ns: int): return 1-it*(Ms+Ns+1)/(Ms*Ns)*1.0

def display_images(img_array: np.ndarray, compressed_img_array: np.ndarray):

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.set_facecolor('#0e1117')

    for i in range(2):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        ax[i].get_xaxis().set_ticks([])
        ax[i].get_yaxis().set_ticks([])

    ax[0].imshow(img_array.astype(np.float64), cmap="gray")

    ax[0].set_title("Original Image", color='white', fontsize=15, fontweight='bold', pad=20)

    ax[1].imshow(compressed_img_array.astype(np.float64), cmap="gray")
    ax[1].set_title("Compressed Image (SVD)", color='white', fontsize=15, fontweight='bold', pad=20)

    st.pyplot(fig)

st.title("SVD Image Compression")

max_value_devide = 10 if 1000 > 100 else 2

def destroy():
    st.session_state["load"].release()
    cv2.destroyAllWindows()
    st.session_state.initialized = False

if "load" not in st.session_state or not st.session_state.initialized:
    cap = cv2.VideoCapture(0)    
    st.session_state["load"] = cap
    st.session_state["init"] = 0
    st.session_state.initialized = True


stop_button = st.button("Stop", on_click=destroy)

frame_placeholder = st.empty() 

n_components = st.slider("Number of components:", min_value=1, max_value=int(1000/max_value_devide), value=max_value_devide, key="slider")  


if "init" not in st.session_state:
    while st.session_state["load"].isOpened() and not stop_button:
        ret, frame = st.session_state["load"].read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

        u, s, v = np.linalg.svd(gray, full_matrices=False)

        compressed_img_array = u[:, :n_components] @ np.diag(s[:n_components]) @ v[:n_components, :]
        frame_placeholder.image((compressed_img_array/255).astype(np.float64), channels="GRAY", clamp=True, use_column_width=True)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or stop_button:
            st.session_state["load"].release()
            cv2.destroyAllWindows()
            break    

st.session_state["init"]  = True

uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:    
    img = cv2.imread(uploaded_file.name)
    img_array = cv2.cvtColor(np.array(img).astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    u, s, v = np.linalg.svd(img_array, full_matrices=False)

    max_value_devide = 10 if min(img_array.shape) > 100 else 2
    
    n_components = st.slider("Number of components:", min_value=1, max_value=int(min(img_array.shape)/max_value_devide), value=max_value_devide)
    compressed_img_array = u[:, :n_components] @ np.diag(s[:n_components]) @ v[:n_components, :]

    st.subheader(f"Compression rate: {(100*compression_rate(5*n_components,u.shape[0],v.shape[0])):.2f}%", )    
    display_images(img_array, compressed_img_array)
