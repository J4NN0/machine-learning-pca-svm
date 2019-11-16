# Support Vector Machines

[Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine) (SVMs) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis.

Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier.

### Classification

Classifying data is a common task in machine learning. Suppose some given data points each belong to one of two classes, and the goal is to decide which class a new data point will be in.

- Linear Classification

    A data point is viewed as a *p-dimensional* vector, and we want to know whether we can separate such points with a *(p-1)-dimensional* hyperplane.

    There are many hyperplanes that might classify the data. One reasonable choice as the best hyperplane is the one that represents the largest separation, or margin, between the two classes.
    
- Non-Linear Classification

    Nonlinear classifiers consist applying the kernel trick to maximum-margin hyperplanes. 
    
    The resulting algorithm is formally similar, except that every dot product is replaced by a nonlinear kernel function. This allows the algorithm to fit the maximum-margin hyperplane in a transformed feature space.

    Some common kernels:
    
    - Polynomial
        - [Homogeneous](https://en.wikipedia.org/wiki/Homogeneous_polynomial)
        - [Inhomogeneous](https://en.wikipedia.org/wiki/Polynomial_kernel)
    - Gaussian [Radial Basis Function](https://en.wikipedia.org/wiki/Radial_basis_function_kernel) (rbf)
    - [Hyperbolic tangent](https://en.wikipedia.org/wiki/Hyperbolic_function)

# Main steps

1. Split data-set into training, validation and test set. But how to choose the parameter?
    - You cannot choose the optimal values based on your test set because of in real problems you won’t have access to it
    - The proper way to evaluate a method is to split the data into 3 parts:
        - Train set: This is used to build up our prediction algorithm. Our algorithm tries to tune itself to the quirks of the training data sets. In this phase we usually create multiple algorithms in order to compare their performances during the Cross-Validation Phase.
        - Validation set: This data set is used to compare the performances of the prediction algorithms that were created based on the training set. We choose the algorithm that has the best performance.
        - Test set: Now we have chosen our preferred prediction algorithm but we don't know yet how it's going to perform on completely unseen real-world data. So, we apply our chosen prediction algorithm on our test set in order to see how it's going to perform so we can have an idea about our algorithm's performance on unseen data.
    - Choose some parameters and train your model on the training set
    - Evaluate the performances on the validation set
    - Once discovered the parameters which work best on the validation set, just apply the same model on the test set

    *Note:* The role of the validation set is to find the optimum for the parameters of the algorithm.

2. Choose the kernel
    - linear
    - rbf
    - ...
3. Choose the C parameter
    - It tells to the SVM optimization how much you want to avoid missclassifying at each training example. For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly. Conversely, a very small value of C will cause the optimizer to look for a larger-margin separating hyperplane, even if that hyperplane missclassifies more points.
    - Briefly, C controls the cost of missclassification on the training data:
        - Small C makes the cost of missclassificaiton low ("soft margin"), thus allowing more of them for the sake of wider "cushion"
        - Large C makes the cost of missclassification high ("hard margin"), thus forcing the algorithm to explain the input data stricter and potentially overfit
    - The goal is to find the balance between "not too strict" and "not too loose". Cross-validation and resampling, along with grid search, are good ways to finding the best C
4. Choose gamma
    - It controls the tradeoff between error due to bias and variance in your model:
        - Small gamma will give you low bias and high variance
        - Large gamma will give you higher bias and low variance

    **Hint**: find the best C and Gamma hyper-parameters using Grid-Search
    
5. Fit the model
6. Plot data and decision boundaries
7. Check the success rate
8. **Note**: The goal of SVM is to find a hyperplane that would leave the widest possible "cushion" between input points from two classes.

# Code implementation

1. Data preparation

    Prepare your data-set, you can easily load iris data if you don't have data to work with 
    
    ```python
    from sklearn import datasets
    
    iris = datasets.load_iris()
    x = iris.data[:, :2]  # take the first two features.
    y = iris.target  # label
    ```
2. Split data into train, test and validation set

    ```python
    from sklearn.model_selection import train_test_split
    
    # 50% training set, 20% validation set, 30% test set
    x_train, x_tmp, y_train, y_tmp = train_test_split(x, y, test_size=0.5, random_state=42)
    x_validate, x_test, y_validate, y_test = train_test_split(x_tmp, y_tmp, test_size=0.4, random_state=42)
    ```
3. Prepare the SVC and fit the model

    Choose the kernel, C and gamma.
    With linear kernel gamma is not used. Then set gamma only if the kernel is not linear.

    ```python
    from sklearn import svm
    
    c = [10**(-3), 10**(- 2), 10**(- 1), 10**0, 10**1, 10**2, 10**3]  # possible values of C
    g = [10**(-9), 10**(-7), 10**(-5), 10**(-3)]  # possible values of gamma
 
    clf = svm.SVC(kernel='rbf', C=c[2], gamma=g[1])  # for example
    clf.fit(x_train, y_train)  # fit the model
    ```
4. Plot data and decision boundaries

    ```python
    import matplotlib.pyplot as plt
 
    # create a mesh to plot in
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    h = (x_max / x_min) / 100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    
    plt.figure('SVC with linear rbf-kernel')  # according to the example
    title = 'C=' + str(c[2]) + ' and gamma=' + str(g[1])  # according to the example
    plt.figure(figsize=(40, 20))
    plt.title(title, fontsize=30)
 
    plt.contourf(xx, yy, z, cmap=plt.get('Paired'), alpha=0.8)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.get('Paired'))
    plt.xlim(xx.min(), xx.max())        

    plt.show()
    plt.close()
    ```
5. Evaluate the method on the validation set

    Two methods. You can do
    
    ```python
    full_prediction = clf.predict(x_validate)
    
    print('[-] Number of mislabeled points out of a total %d points: %d'
          % (len(y_validate), (y_validate != full_prediction).sum()))
    succ_rate = (100 * (len(y_validate) - (y_validate != full_prediction).sum())) / len(y_validate)
    print('[-] With am accuracy of %.3f%s'
          % (succ_rate, '%'))
    ```
    Or
    
    ```python
    clf_predictions = clf.predict(x_validate)
    print("Accuracy: {}%".format(clf.score(x_validate, y_validate) * 100 ))
    ```
6. Choose the best pair of *C* and *gamma* and evaluate test set

    ```python
    clf = svm.SVC(kernel='rbf', C=best_c, gamma=best_gamma)
    clf.fit(x_train, y_train)
 
    full_prediction = clf.predict(x_test)
 
    print('[-] Number of mislabeled points out of a total %d points: %d'
          % (len(y_test), (y_test != full_prediction).sum()))
    print('[-] With a success rate of %.3f%s'
          % ((100 * (len(y_test) - (y_test != full_prediction).sum())) / len(y_test), '%'))
    ```
    You can also plot 

## K-Fold Cross Validation

K-Fold Cross Validation is useful when the training set is small. In this case the validation set is a subset of the training data and is data we can no longer use to train our model.

1. Data preparation

    ```python
    from sklearn import datasets
    
    iris = datasets.load_iris()
    x = iris.data[:, :2]  # take the first two features.
    y = iris.target  # label
    ```

2. Split data into train and test set

    ```python
    from sklearn.model_selection import train_test_split
 
    # 70% training set, 30% test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    ```
3. Divide training data into k equal parts

    ```python
    from sklearn.model_selection import KFold
 
    k_fold = KFold(n_splits=5)  # performing 5-Fold Cross Validation
    ```
    
    **Remember**: too small of a validation set is not very informative, too big and you lose too much training data.

4. Do k rounds of validation

    ```python
    for id_train, id_test in k_fold.split(x_train):
        svc = svm.SVC(kernel='rbf', C=c, gamma=g)  # choose a valid C and gamma
        score = svc.fit(x_train[id_train], y_train[id_train]).score(x_train[id_test], y_train[id_test])
        print('With C=' + str(c) + ' and gamma=' + str(g) + ' avg=' + str(score))
    ```
    
## 2D heatmap validation accuracy

You can plot validation accuracy on 2D heatmap

```python
import matplotlib.pyplot as plt

plt.title('Validation Accuracy')
plt.xlabel('gamma')
plt.ylabel('C')
plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()
```
Where **a** si a matrix in which rows are C values and columns are gamma values, each cell is the accuracy for the corresponding C and gamma.

# SVM application

- Classification of image
- Recognize hand-written characters
- Classify proteins

# Pros and Cons

- Pros  
    - Guaranteed Optimality: due to the nature of Convex Optimization, the solution is guaranteed to be the global minimum not a local minimum.
    - It uses the kernel trick, so you can build in expert knowledge about the problem via engineering the kernel.

- Cons
    - Kernel models can be quite sensitive to over-fitting the model selection criterion.
    - In Natural Language Processing, structured representations of text yield better performances.
 
# Utility

- [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
- [SVM](https://scikit-learn.org/stable/modules/svm.html)
- [C-Support Vector Classification](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [K-Fold](https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html)
- [Cmap](https://matplotlib.org/examples/color/colormaps_reference.html)
