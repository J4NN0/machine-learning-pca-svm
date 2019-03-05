# Introduction

PCA folder contains the PCA applied on images. It is showed what happens if different principal components (PC) are chosen as basis for images representation and classification. Then, the Naive Bayes Classifier has been choosen and applied in order to classify the image.

# General requirements

- numpy

        pip install numpy

- Pillow

        pip install Pillow

- scikit-learn

        pip install -U scikit-learn

- Matplotlib

        sudo pip install matplotlib
        
### Fix

If using 'sklearn' you get the following error

> DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses import imp

Check file 'cloudpickle.py' and delete row

     imports imp