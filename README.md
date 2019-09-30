# machine-learning-pca

PCA folder contains the PCA applied on images. It is shown what happens if different principal components (PC) are chosen as basis for images representation and classification. Then, the Naive Bayes Classifier has been choosen and applied in order to classify the image.

# PCA overview

### Image reconstruction

<img width="1022" alt="img_reconstruction" src="https://user-images.githubusercontent.com/25306548/65880762-9ff6e900-e392-11e9-8c4f-7c7ee2c02421.png">

### PCA visualization

<img width="714" alt="lenna" src="https://user-images.githubusercontent.com/25306548/65881390-9e79f080-e393-11e9-89ca-452c8eb5cb92.png">

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
