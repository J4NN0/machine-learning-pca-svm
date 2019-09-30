# machine-learning-pca

PCA folder contains the PCA applied on images. It is shown what happens if different principal components (PC) are chosen as basis for images representation and classification. Then, the Naive Bayes Classifier has been choosen and applied in order to classify the image.

# PCA overview

### Image reconstruction

Example result of PCA application on images: in the 2-PC and 6-PC is possible to see (with a little attention) the silhouette (shape) of a dog. Instead, in the 60-PC the silhouette is more evident. The more are the number of PC the more easier to see becomes the solhouette of the dog. The last 6-PC, as expected, are really bad and it is not possible to understand nothing.

<img width="1022" alt="img_reconstruction" src="https://user-images.githubusercontent.com/25306548/65880762-9ff6e900-e392-11e9-8c4f-7c7ee2c02421.png">

### PC visualization

Each color is a different type of subjetc: blue -> dogs, green -> guitar, red -> houses and yellow -> people. 
The higher is the number of PC the more is the number informations that are brought. In the figure below is showed how, trough PC visualization, is possible to distinguish the different classes.

<img width="976" alt="pc_visualization" src="https://user-images.githubusercontent.com/25306548/65881812-442d5f80-e394-11e9-870c-d2a860a1e366.png">

**I suggest to read the [documentation](https://github.com/J4NN0/machine-learning-pca/blob/master/doc/pca_report.pdf) in order to have a more in-depth explenation.**

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
