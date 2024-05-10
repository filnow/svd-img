import numpy as np


def calculSigma(M): 
    if (np.size(np.dot(M, M.T)) > np.size(np.dot(M.T, M))): 
        newM = np.dot(M.conj().T, M) 
    else: 
        newM = np.dot(M, M.conj().T) 
        
    eigenvalues, _ = np.linalg.eig(newM) 
    eigenvalues = np.sqrt(eigenvalues) 
    eigenvalues = np.sort(eigenvalues)
    
    return eigenvalues[::-1] 

def calculVt(M): 
    B = np.dot(M.conj().T, M)
        
    eigenvalues, eigenvectors = np.linalg.eig(B) 
    ncols = np.argsort(eigenvalues)[::-1] 
    
    return eigenvectors[:,ncols].conj().T 

def calculU(M): 
    B = np.dot(M, M.conj().T)      
    eigenvalues, eigenvectors = np.linalg.eig(B) 
    ncols = np.argsort(eigenvalues)[::-1] 
    
    return eigenvectors[:,ncols] 

def svd(M):
    U = calculU(M)
    Sigma = calculSigma(M)
    Vt = calculVt(M)
    
    return U, Sigma, Vt


if __name__ == '__main__':
    #NOTE: resource https://acme.byu.edu/00000181-a729-d778-a18f-bf6b263d0000/the-svd-and-image-compression-pdf
    A = np.array([[4,2,0],[1,5,6]])

    print("\nInput matrix A:\n", A)
    
    npU, npsigma, npV_T = np.linalg.svd(A)

    print("\nValues for U and Sigma and V_T from numpy\n")
    
    print(f"U array:\n {npU}\n")
    print(f"Sigma array:\n {npsigma}\n")
    print(f"V_T array:\n {npV_T}\n")

    print("------------------------------------------------------\n")

    print("Values for U and Sigma and V_T from our implementation\n")

    U, sigma, V_T = svd(A)

    print(f"U array:\n {U}\n")
    print(f"Sigma array:\n {sigma}\n")
    print(f"V_T array:\n {V_T}\n")
