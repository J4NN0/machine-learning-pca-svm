# Machine Learning

### PCA & Naive Bayes Classifier

It is shown what happens if different principal components (PC) are chosen as basis for images representation and classification. Then, the Naive Bayes Classifier has been choosen and applied in order to classify the image.

### SVM

Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier.

# Overview

### Image reconstruction

Example result of PCA application on images: in the 2-PC and 6-PC is possible to see (with a little attention) the silhouette (shape) of a dog. Instead, in the 60-PC the silhouette is more evident. The more are the number of PC the more easier to see becomes the solhouette of the dog. The last 6-PC, as expected, are really bad and it is not possible to understand nothing.

<img width="1022" alt="img_reconstruction" src="https://user-images.githubusercontent.com/25306548/65880762-9ff6e900-e392-11e9-8c4f-7c7ee2c02421.png">

### PC visualization

Each color is a different type of subjetc: blue -> dogs, green -> guitar, red -> houses and yellow -> people. 
The higher is the number of PC the more is the number informations that are brought. In the figure below is showed how, trough PC visualization, is possible to distinguish the different classes.

<img width="976" alt="pc_visualization" src="https://user-images.githubusercontent.com/25306548/65881812-442d5f80-e394-11e9-870c-d2a860a1e366.png">

### Naive Bayes Classifier

The classifier is able to predict, given an observation of an input, a probability distribution over a set of classes, rather than only outputting the most likely class that the observation should belong to. After splitting the data-set into train and test set, I used the Naive Bayed Classifier in several cases and checked the respective accuracy.

<img width="445" alt="classifier" src="https://user-images.githubusercontent.com/25306548/65883630-b5badd00-e397-11e9-8123-637a84a27ea2.png">

*I suggest to read the [documentation](https://github.com/J4NN0/machine-learning-pca/blob/master/doc/pca_report.pdf) in order to have a more in-depth explenation. Also a [demonstration video](https://www.youtube.com/watch?v=6ltDO_momlI) is available on youtube.*

### SVM Classification

An example of two different graphs in data classifciation using the rbf (Radial Basis Function) kernel.

- Data and decision boundaries

     <img width="625" alt="test_set" src="https://user-images.githubusercontent.com/25306548/68994274-67976580-0881-11ea-8fba-0bc973d53f57.png">

- This is the validation accuracy

     <img width="449" alt="validation_accuracy" src="https://user-images.githubusercontent.com/25306548/68994285-7da52600-0881-11ea-9072-140831ab63df.png">

*Checkout the [documentation](https://github.com/J4NN0/machine-learning-pca-svm/blob/master/doc/svm_report.pdf) for more infos.*

# General requirements

- numpy

        pip install numpy

- Pillow

        pip install Pillow

- scikit-learn

        pip install -U scikit-learn

- Matplotlib

        sudo pip install matplotlib
        
### Issues

If using 'sklearn' you get the following error

> DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses import imp

Check file 'cloudpickle.py' and delete row

     imports imp
