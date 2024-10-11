import scipy
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    x= x - np.mean(x, axis=0)
    return x

def get_covariance(dataset):
    x = dataset
    dot = np.dot(np.transpose(x),x)
    cov = (1/(len(x)-1))*dot
    return cov

def get_eig(S, m):
    matrixSize = len(S)
    evals, evecs = eigh(S, subset_by_index = [matrixSize-m,matrixSize-1])
    evals = evals[::-1]
    evecs = evecs[:,::-1]
    evals = np.diag(evals)
    return evals, evecs

def get_eig_prop(S, prop):
    lower_bound = S.trace() * prop
    evals, evecs = eigh(S, subset_by_value=[lower_bound,np.inf])
    evals = evals[::-1]
    evecs = evecs[:, ::-1]
    evals = np.diag(evals)
    return evals, evecs

def project_image(image, U):
    x = image
    uT = np.transpose(U)
    aij = np.dot(uT,x)
    return np.dot(U,aij)


def display_image(orig, proj):
    rorig = orig.reshape(64,64)
    rproj = proj.reshape(64, 64)
    rproj = np.transpose(rproj)
    fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    a1 = ax1.imshow(rorig, aspect="equal")
    ax1.set_title("Original")
    a2 = ax2.imshow(rproj, aspect="equal")
    ax2.set_title('Projection')

    fig.colorbar(a1,ax=ax1, location='right')
    fig.colorbar(a2,ax=ax2, location='right')

    return fig, ax1, ax2

def main():
    x = load_and_center_dataset('Iris_64x64.npy')
    S = get_covariance(x)
    Lambda, U = get_eig(S,2)
    print(Lambda)
    print(U)
    #Lambda, U = get_eig_prop(S, 0.07)
    #print(Lambda)
    #print(U)
    projection = project_image(x[50], U)
    fig,ax1,ax2 = display_image(x[50],projection)
    plt.show()




if __name__ == "__main__":
    main()