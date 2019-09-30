from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.cm import *


def print_single_pca(ncomp, matrix, title):
    plt.title(title)

    for i in range(0, ncomp, +2):
        plt.scatter(matrix[i, :], matrix[i, :], c='r')


def main():
    x = []

    print('Opening photo')
    img_data = np.asarray(Image.open('photos/lena512color.tiff'))
    # img_data.shape
    x = img_data.reshape(512, -1)

    # Standardizing matrix: mean = 0 and variance = 1
    print('Standardizing matrix')
    mean = np.mean(x)  # Mean of X
    std = np.std(x)  # Standard deviation of X
    x = (x - mean) / std

    # Using PCA to extract two principal components from sample covariance of X
    # fit_transform: Fit the model with X and apply the dimensionality reduction on X
    # (i.e. in this case X_t will be n_photo*2)
    print('Calculating first 50-PC')
    n_comp = 50
    pca = PCA(n_comp)
    x_t = pca.fit_transform(x)
    print(pca.explained_variance_ratio_.cumsum())

    # Visualizing data
    print('Visualizing data of image')
    plt.figure('Lenna data')
    print_single_pca(n_comp, x_t, '50-PC Components')
    plt.show()
    plt.close()

    print('Producing images')
    plt.figure('Lenna photos', figsize=(30, 8))
    # Original image
    plt.subplot(1, 2, 1)
    x = (x * std) + mean  # De-Standardizing matrix
    original = np.reshape(x, (512, 512, -1)).astype(int)
    plt.imshow(original, interpolation='nearest')
    plt.title('Original', fontsize=30)
    # N-PC image
    plt.subplot(1, 2, 2)
    approx = pca.inverse_transform(x_t)
    approx = (approx * std) + mean
    img_components = np.reshape(approx, (512, 512, -1)).astype(int)
    plt.imshow(img_components, interpolation='nearest')
    plt.title(str(n_comp) + '-PC', fontsize=30)
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
