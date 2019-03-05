# Naive Bayes Classifier

[Naive Bayes classifiers](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) are a family of simple "probabilistic classifier": a [probabilistic classifier](https://en.wikipedia.org/wiki/Probabilistic_classification) is a classifier that is able to predict, given an observation of an input, a probability distribution over a set of classes, rather than only outputting the most likely class that the observation should belong to.

The Naive Bayes classifier is based on applying Bayes' theorem with strong (naive) independence assumptions between the features.

[Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Introduction) is a simple technique for constructing classifiers and it is easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods.

### Probabilistic model

Abstractly, [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Probabilistic_model) is a conditional probability model: given a problem instance to be classified, represented by a vector **x** representing some *n featuers* (independent variables) it assigns to this instance probabilities:

> ![photo](https://wikimedia.org/api/rest_v1/media/math/render/svg/c6ebabd72c70e181cf901c415a87636a6474f139)

Using [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem), the conditional probability can be decomposed as

> ![photo](https://wikimedia.org/api/rest_v1/media/math/srender/svg/52bd0ca5938da89d7f9bf388dc7edcbd546c118e)

In which:
- P(C|x) is the *posterior* probability for class C given predictor **x**
- P(C) is the *prior* probability of class C
- P(x|C) is the is the *likelihood* which is the probability of predictor given class.
- P(x) is the *prior* probability of predictor

In this way the above equation can be written as:

> ![photo](https://wikimedia.org/api/rest_v1/media/math/render/svg/d0d9f596ba491384422716b01dbe74472060d0d7)

#### Constructing a classifier from the probability model

The naive Bayes classifier combines this model with a [decision rule](https://en.wikipedia.org/wiki/Decision_rule): a function which maps an observation to an appropriate action. 
One common rule is to pick the hypothesis that is most probable; this is known as the [maximum a posteriori](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) or MAP decision rule. 
The corresponding classifier, a Bayes classifier, is the function that assigns a class label 

> ![photo](https://wikimedia.org/api/rest_v1/media/math/render/svg/6fe719eda4ce62ee2f2104455abc5233fdf69e01)

for some k as follows:

> ![photo](https://wikimedia.org/api/rest_v1/media/math/render/svg/5ed52009429e5f3028302427a067822fdfc58059)

where:
- *y* is a predicted lable
- *k* is the number of classes
- **x** are examples

### Event models

The assumptions on distributions of features are called the event model of the Naive Bayes classifier.

There different Naive Bayes models:
- [Gaussian Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes): it is used in classification and it assumes that features follow a normal distribution.
- [Multinomial Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_naive_Bayes): it is used for discrete counts.
- [Bernoulli Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Bernoulli_naive_Bayes): the binomial model is useful if your feature vectors are binary.

Based on your data set, choose any of these models.

# Main steps

- Split data-set into training and test set
- Choose a Naive Bayes model
- Train the data
- Get a prediction of a specific data
- Check if that prediction is equal to the corresponding data of the test set

# Code implementation

1. Data preparation

    Read your data-set and create a matrix in which each row is an 1D array associated to an image.
    Then prepare and array which will contain the associated label for each class: for example if you the matrix contains from 0 to n the class 'dog' the array will contain from 0 to n the label 'dog' and so on.

2. Split examples in train and test set
    
    ```python
    from sklearn.model_selection import train_test_split
 
    x_train, x_test, y_train, y_test = train_test_split(matrix, label, test_size=0.33, random_state=42)
    ```
        
3. Choose the Naive Bayes classifier

    ```python
    from sklearn.naive_bayes import GaussianNB
 
    clf = GaussianNB()
    ```   
    In this case I opted for Gaussian class-conditional distribution
         
4. Train data

    ```python
    clf.fit(x_train, y_train)
    ```
        
5. Choose an 'item' from the training set and get a prediction for that 'item'

    ```python
    r = random.randint(0, len(x_test) - 1)  # Choose a random item in the data-set
    y_predict = clf.predict([x_test[r]])
    ```
        
    Function will return label of that item
        
6. Compare predicted label with the exacted once. If they are equals the prediction is good otherwise not. Just print it and see it

    ```python
    print(y_predict)
    print(y_test[r])
    ```
        
    If the data are good, in the 75% (more or less) of cases the prediction will be correct

# Naive Bayes application

- Real time prediction: it's fast and so it can be used to make predictions in real time
- Multi class prediction: predict the probability of multiple classes of target variable.
- Text classification
    - Spam Filtering: identify spam e-mail
    - Sentiment Analysis: in social media analysis, to identify positive and negative customer sentiments

# Pros and cons

- Pros
    - It is fast to predict class of test data set. It also perform well in multi-class prediction.
    - When assumption of independence holds, a Naive Bayes classifier performs better compare to other models like logistic regression and you need less training data.
    
- Cons
    - If categorical variable has a category (in test data set), which was not observed in training data set, then model will assign a 0 (zero) probability and will be unable to make a prediction. This is often known as “Zero Frequency”.
    - Another limitation of Naive Bayes is the assumption of independent predictors. In real life, it is almost impossible that we get a set of predictors which are completely independent.

# Utility

- [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
- [Train test split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)