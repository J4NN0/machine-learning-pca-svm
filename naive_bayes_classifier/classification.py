from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import glob
import random

# Global variables
# Total number of photo inside each folder
ndog = len(glob.glob('photos/dog/*.jpg'))
nguitar = len(glob.glob('photos/guitar/*.jpg'))
nhouse = len(glob.glob('photos/house/*.jpg'))
nperson = len(glob.glob('photos/person/*.jpg'))


def open_img(path, matrix):
    for filename in glob.glob(path):
        # 3-D array from an image
        img_data = np.asarray(Image.open(filename))
        # Converting image into 154587-dimensional vector and preparing matrix n_photo*154587
        matrix.append(img_data.ravel())

    return matrix


def training(matrix, label):
    print('Splitting matrix into random train and test subsets')
    x_train, x_test, y_train, y_test = train_test_split(matrix, label, test_size=0.33, random_state=42)

    clf = GaussianNB()  # Gaussian Naive Bayes
    clf.fit(x_train, y_train)  # Fit Gaussian Naive Bayes according to x, y training set
    r = random.randint(0, len(x_test) - 1)  # Choose a random item in the data-set
    y_predict = clf.predict([x_test[r]])  # Get prediction of that item
    full_prediction = clf.predict(x_test)  # Get prediction of all item

    print('Prediction is: ')
    print(y_predict)
    print('Test value is: ')
    print(y_test[r])

    print('[!] Number of mislabeled points out of a total %d points: %d'
          % (len(y_test), (y_test != full_prediction).sum()))
    print('[+] With a success rate of %.3f%s'
          % ((100 * (len(y_test) - (y_test != full_prediction).sum()) / len(y_test)), '%'))


def main():
    x = []
    y = []

    # Building label of y
    for i in range(0, ndog):
        y.append('dog')
    for i in range(ndog, ndog + nguitar):
        y.append('guitar')
    for i in range(ndog + nguitar, ndog + nguitar + nhouse):
        y.append('house')
    for i in range(ndog + nguitar + nhouse, ndog + nguitar + nhouse + nperson):
        y.append('person')

    print('Calculating matrix Nx154587')
    x = open_img('PACS_homework/dog/*.jpg', x)
    x = open_img('PACS_homework/guitar/*.jpg', x)
    x = open_img('PACS_homework/house/*.jpg', x)
    x = open_img('PACS_homework/person/*.jpg', x)

    print('[-] Training data')
    # training(x, y)

    # Standardizing matrix: mean = 0 and variance = 1
    print('Standardizing matrix')
    x = (x - np.mean(x)) / np.std(x)

    print('Calculating first 4-PC')
    x_t = PCA(2).fit_transform(x)

    print('[-] Training data with 4-PC')
    training(x_t, y)

    # print('[-] Training with first and second of 4-PC')
    # training(x_t[:, 0:2], y)
    # print('[-] Training with third and fourth of 4-pc')
    # training(x_t[:, 2:4], y)


if __name__ == "__main__":
    main()
