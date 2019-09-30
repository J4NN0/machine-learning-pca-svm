from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.cm import *

import glob

# Global variables
# Total number of photo inside each folder
ndog = len(glob.glob('../photos/dog/*.jpg'))
nguitar = len(glob.glob('../photos/guitar/*.jpg'))
nhouse = len(glob.glob('../photos/house/*.jpg'))
nperson = len(glob.glob('../photos/person/*.jpg'))


def open_imgs_set(path, matrix):
    for filename in glob.glob(path):
        # 3-D array from an image
        img_data_3d = np.asarray(Image.open(filename))
        # From 3-D array to 1-D array: converting single image into 1-D array of size n (int this case n=154587)
        img_data_1d = img_data_3d.ravel()
        #  Preparing matrix n_photo*154587
        matrix.append(img_data_1d)

    return matrix


def print_pca(ncomp, matrix, title):
    global ndog, nguitar, nhouse, nperson

    # Ordinal label of the images (colours)
    y = ['b', 'g', 'r', 'y']

    plt.title(title, fontsize=30)

    # Diving with different colors the different data-set
    for i in range(0, ncomp-1):
        plt.scatter(matrix[0:ndog, i], matrix[0:ndog, i+1], c=y[0])
        plt.scatter(matrix[ndog:ndog+nguitar, i], matrix[ndog:ndog+nguitar, i+1], c=y[1])
        plt.scatter(matrix[ndog+nguitar:ndog+nguitar+nhouse, i],
                    matrix[ndog+nguitar:ndog+nguitar+nhouse, i+1], c=y[2])
        plt.scatter(matrix[ndog+nguitar+nhouse:ndog+nguitar+nhouse+nperson, i],
                    matrix[ndog+nguitar+nhouse:ndog+nguitar+nhouse+nperson, i+1], c=y[3])


def print_single_pca(ncomp, matrix, photo_id, title):
    color = 'k'

    plt.title(title)

    for i in range(0, ncomp, +2):
        plt.scatter(matrix[photo_id, i], matrix[photo_id, i+1], c=color)


def last_n_pca(matrix, lpc):
    # PCA from scratch
    cov = np.cov(matrix)  # covariance matrix
    eigval, eigvec = np.linalg.eig(cov)  # computing the eigenvalues and eigenvectors
    # Make a list of (eigenvalue, eigenvectors) tuples
    eig_pairs = [(np.abs(eigval[i]), eigvec[:, i]) for i in range(len(eigval))]
    # Sort the (eigenvalues, eigenvectors) tuples from low to high (sort from high to low for the first n-PC)
    eig_pairs.sort(key=lambda x: x[0], reverse=False)  # reverse=True means the first n-PC
    matrix_w = np.hstack([eig_pairs[i][1].reshape(len(matrix), 1) for i in range(lpc)])

    return matrix_w


def main():
    x = []

    print('Calculating matrix Nx154587')
    x = open_imgs_set('PACS_homework/dog/*.jpg', x)
    x = open_imgs_set('PACS_homework/guitar/*.jpg', x)
    x = open_imgs_set('PACS_homework/house/*.jpg', x)
    x = open_imgs_set('PACS_homework/person/*.jpg', x)

    # Standardizing matrix: mean = 0 and variance = 1
    print('Standardizing matrix')
    mean = np.mean(x)  # Mean of X
    std = np.std(x)  # Standard deviation of X
    x = (x - mean) / std

    """""
    pca11 = PCA(11)
    y = ['b', 'g', 'r', 'y']
    x_t_11 = pca11.fit_transform(x)

    plt.figure('PC', figsize=(30, 8))

    plt.subplot(131)
    plt.title('2-PC', fontsize=30)
    for i in range(0, 2-1):
        plt.scatter(x_t_11[0:ndog, i], x_t_11[0:ndog, i+1], c=y[0])
        plt.scatter(x_t_11[ndog:ndog+nguitar, i], x_t_11[ndog:ndog+nguitar, i+1], c=y[1])
        plt.scatter(x_t_11[ndog+nguitar:ndog+nguitar+nhouse, i],
                    x_t_11[ndog+nguitar:ndog+nguitar+nhouse, i+1], c=y[2])
        plt.scatter(x_t_11[ndog+nguitar+nhouse:ndog+nguitar+nhouse+nperson, i],
                    x_t_11[ndog+nguitar+nhouse:ndog+nguitar+nhouse+nperson, i+1], c=y[3])

    plt.subplot(132)
    plt.title('3&4-PC', fontsize=30)
    for i in range(2, 4-1):
        plt.scatter(x_t_11[0:ndog, i], x_t_11[0:ndog, i+1], c=y[0])
        plt.scatter(x_t_11[ndog:ndog+nguitar, i], x_t_11[ndog:ndog+nguitar, i+1], c=y[1])
        plt.scatter(x_t_11[ndog+nguitar:ndog+nguitar+nhouse, i],
                    x_t_11[ndog+nguitar:ndog+nguitar+nhouse, i+1], c=y[2])
        plt.scatter(x_t_11[ndog+nguitar+nhouse:ndog+nguitar+nhouse+nperson, i],
                    x_t_11[ndog+nguitar+nhouse:ndog+nguitar+nhouse+nperson, i+1], c=y[3])

    plt.subplot(133)
    plt.title('10&11-PC', fontsize=30)
    for i in range(9, 11-1):
        plt.scatter(x_t_11[0:ndog, i], x_t_11[0:ndog, i+1], c=y[0])
        plt.scatter(x_t_11[ndog:ndog+nguitar, i], x_t_11[ndog:ndog+nguitar, i+1], c=y[1])
        plt.scatter(x_t_11[ndog+nguitar:ndog+nguitar+nhouse, i],
                    x_t_11[ndog+nguitar:ndog+nguitar+nhouse, i+1], c=y[2])
        plt.scatter(x_t_11[ndog+nguitar+nhouse:ndog+nguitar+nhouse+nperson, i],
                    x_t_11[ndog+nguitar+nhouse:ndog+nguitar+nhouse+nperson, i+1], c=y[3])

    plt.show()
    plt.close()
    return
    """


    # Using PCA to extract two principal components from sample covariance of X
    # fit_transform: Fit the model with X and apply the dimensionality reduction on X
    # (i.e. in this case X_t will be n_photo*2)
    print('Calculating first 2-PC')
    pca2 = PCA(2)
    x_t_2 = pca2.fit_transform(x)
    print(pca2.explained_variance_ratio_.cumsum())

    # From the whole data-set calculating the first 2 (done above), 6 and 60 PC
    print('Calculating first 6-PC')
    pca6 = PCA(6)
    x_t_6 = pca6.fit_transform(x)
    print(pca6.explained_variance_ratio_.cumsum())
    print('Calculating first 60-PC')
    pca60 = PCA(60)
    x_t_60 = pca60.fit_transform(x)
    print(pca60.explained_variance_ratio_.cumsum())

    # From the whole data-set calculating the last 6 PC
    print('Calculating last 6-PC')
    x_t_last6 = last_n_pca(x, 6)

    # Visualizing data
    print('Visualizing whole data-set')
    plt.figure('Whole data-set')
    plt.subplot(221)
    print_pca(2, x_t_2, '2-PC')
    plt.subplot(222)
    print_pca(6, x_t_6, '6-PC')
    plt.subplot(223)
    print_pca(60, x_t_60, '60-PC')
    plt.subplot(224)
    print_pca(6, x_t_last6, 'Last 6-PC')
    plt.show()
    plt.close()

    photo_id = 120

    print('Visualizing data of image n: ' + str(photo_id))
    plt.figure('Photo N ' + str(photo_id))
    plt.subplot(221)
    print_single_pca(2, x_t_2, photo_id, '2-PC Components')
    plt.subplot(222)
    print_single_pca(6, x_t_6, photo_id, '6-PC Components')
    plt.subplot(223)
    print_single_pca(60, x_t_60, photo_id, '60-PC Components')
    plt.subplot(224)
    print_single_pca(6, x_t_last6, photo_id, 'Last 6-PC Components')
    plt.show()
    plt.close()

    print('Producing images. ID: ' + str(photo_id))
    plt.figure('Image', figsize=(30, 8))
    # Original image
    plt.subplot(1, 5, 1)
    x = (x * std) + mean  # De-Standardizing matrix
    original = np.reshape(x[photo_id], (227, 227, 3)).astype(int)
    plt.imshow(original, interpolation='nearest')
    plt.title('Original', fontsize=30)
    # 2-PC image
    plt.subplot(1, 5, 2)
    approx_2 = pca2.inverse_transform(x_t_2)
    approx_2 = (approx_2 * std) + mean
    img_2_components = np.reshape(approx_2[photo_id], (227, 227, 3)).astype(int)
    plt.imshow(img_2_components, interpolation='nearest')
    plt.title('2-PC', fontsize=30)
    # 6-PC image
    plt.subplot(1, 5, 3)
    approx_6 = pca6.inverse_transform(x_t_6)
    approx_6 = (approx_6 * std) + mean
    img_6_components = np.reshape(approx_6[photo_id], (227, 227, 3)).astype(int)
    plt.imshow(img_6_components, interpolation='nearest')
    plt.title('6-PC', fontsize=30)
    # 60-PC image
    plt.subplot(1, 5, 4)
    approx_60 = pca60.inverse_transform(x_t_60)
    approx_60 = (approx_60 * std) + mean
    img_60_components = np.reshape(approx_60[photo_id], (227, 227, 3)).astype(int)
    plt.imshow(img_60_components, interpolation='nearest')
    plt.title('60-PC', fontsize=30)
    # Last 6-PC image
    plt.subplot(1, 5, 5)
    approx_last6 = pca6.inverse_transform(x_t_last6)
    approx_last6 = (approx_last6 * std) + mean
    img_last6_components = np.reshape(approx_last6[0], (227, 227, 3)).astype(int)
    plt.imshow(img_last6_components, interpolation='nearest')
    plt.title('Last 6-PC', fontsize=30)
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
