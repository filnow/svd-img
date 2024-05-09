import cv2
import numpy as np


cap = cv2.VideoCapture(0)
n_components = 1

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    u, s, v = np.linalg.svd(gray, full_matrices=False)
    
    compressed_img_array = u[:, :n_components] @ np.diag(s[:n_components]) @ v[:n_components, :]
    cv2.imshow("compressed", (compressed_img_array/255).astype(np.float64))
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        n_components += 10
        print(f"Number of components increased to {n_components}")
    
    elif key == 8:
        if n_components > 1:
            n_components -= 10
        else:
            n_components = 1
        print(f"Number of components decreased to {n_components}")

    if key == ord('q'):
        break 