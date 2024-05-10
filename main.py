import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter


def compression_rate(it: int, Ms: int, Ns: int) -> float: 
    return 1-it*(Ms+Ns+1)/(Ms*Ns)*1.0

def display_images(img_array: np.ndarray, 
                   compressed_img_array: np.ndarray) -> None:
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

def svd(img_array: np.ndarray, n_components: int, 
        scall_func: dict, option: str="None") -> np.ndarray:
    u, s, v = np.linalg.svd(img_array, full_matrices=False)
    s = scall_func[option](s)
    compressed_img_array = u[:, :n_components] @ np.diag(s[:n_components]) @ v[:n_components, :]
    return compressed_img_array

def destroy() -> None:
    st.session_state["load"].release()
    cv2.destroyAllWindows()
    st.session_state.initialized = False
    slider_placeholder.empty()


if __name__ == "__main__":    
    # Dictionary with the functions to scall the singular values
    scall_func = {
                    'None': lambda x: x, 
                    'Log': lambda x: np.log(x+1), 
                    'Sinusoidal': lambda x: np.sin(x),
                    'Gaussian Filter': lambda x: gaussian_filter(x, sigma=1)
                }
        
    st.title("SVD Image Compression")
    stop_button = st.button("Stop video", on_click=destroy)
    
    slider_placeholder = st.empty() 
    frame_placeholder = st.empty() 

    n_components = slider_placeholder.slider("Number of components:", 
                                min_value=1, max_value=100, value=5, key="slider") 

    # Load the webcam
    if "load" not in st.session_state or not st.session_state.initialized:
        cap = cv2.VideoCapture(0)    
        st.session_state["load"] = cap
        st.session_state.initialized = True
        
    # Display the webcam with SVD compression
    if "init" not in st.session_state:
        while st.session_state["load"].isOpened() and not stop_button:
            ret, frame = st.session_state["load"].read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

            u, s, v = np.linalg.svd(gray, full_matrices=False)

            compressed_img_array = u[:, :n_components] @ np.diag(s[:n_components]) @ v[:n_components, :]
            
            frame_placeholder.image((compressed_img_array/255).astype(np.float32), 
                                    channels="GRAY", clamp=True, use_column_width=True)

            if cv2.waitKey(1) & 0xFF == ord('q') or stop_button:
                st.session_state["load"].release()
                cv2.destroyAllWindows()
                break    

    st.session_state["init"]  = True

    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

    # Display the image with SVD compression
    if uploaded_file is not None:
        img_array = np.array(cv2.imread("./images/" + uploaded_file.name)).astype(np.float32)       
        
        gray_checkbox = st.checkbox("Gray scale", False)

        # Number of components for the SVD
        n = st.slider("Number of components:", min_value=1, 
                        max_value=int(min(img_array.shape[:2])/10), value=10, key="slider_img")
        
        # Function to scall the singular values
        option = st.selectbox("Function to scall singular values:", (scall_func.keys()), key="option")

        if gray_checkbox:
            # Gray scale image
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY).astype(np.float32)
            compressed_img_array = svd(img_array, n, scall_func, option)
        else:
            # RGB image
            img_array = cv2.cvtColor(img_array/255, cv2.COLOR_BGR2RGB)
            red, green, blue = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            compressed_img_array = np.zeros((np.array(img_array).shape[0], np.array(img_array).shape[1], 3))

            for i, channel in enumerate([red, green, blue]):
                compressed_img_array[:, :, i] = svd(channel, n, scall_func, option)
            
        st.subheader(f"Compression rate: {(100*compression_rate(5*n, img_array.shape[0], img_array.shape[1])):.2f}%")    
        
        display_images(img_array, compressed_img_array)
