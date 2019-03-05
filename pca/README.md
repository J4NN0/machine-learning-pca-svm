# PCA

[Principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components.

If there are *n* observations with *p* variables, then the number of distinct principal components is *min(n-1, p)*. This transformation is defined in such a way that the first principal component has the largest possible variance.

PCA is mostly used as a tool in exploratory data analysis and for making predictive models.

For more [details](https://en.wikipedia.org/wiki/Principal_component_analysis#Details).

### The idea

Given data points into d-dimensional space, project them into a lower dimensional space while preserving as much information as possible.
In particular, we need to choose projection that minimizes squares error in reconstructing original data.

Principal Components are points in the direction of the largest variance. Each subsequent principal components:
- is orthogonal to the previous ones
- points in the direction of the largest variance of the residual subspace

# Main steps

- Calculate the covariance matrix
- Compute the eigenvalues and eigenvectors
- Sort the (eigenvalues, eigenvectors) tuples from high to low in order to have the largest possible variance

# Code implementation

1. Data preparation

    Read your data-set and create a matrix in which each row is an 1D array associated to an image.
    You can read an huge data-set or a single image. The more are the data the more accurate the final result will be.
    
    1. Data preparation (bis)
    
        When you use 
        
        ```python
        from PIL import Image
        import numpy as np
    
        img_data_3d = np.asarray(Image.open(filename))
        ```
        The function will return a 3D array of the corresponding image.
        You have to scale back the 3D array into 2D array or 1D array.
        
        If you want to know the dimension of the returned array
        
        ```python
        img_data_3d.shape
        ```
        It will return respectively rows, columns and 'depth'. In my case the used images are (227, 227, 3)
        
        If you want to scale back into 1-dimension
        ```python
        img_data_1d = img_data_3d.ravel()
        ```
        If you want to scale back into 2-dimension
        ```python
        img_data_2d = img_data_3d.reshape(row_size, -1)
        ```
        In which *row_size* is the size of the rows got by ```img_data_3d.shape```
        
        I suggest you to scale back into 1D if you have a lot of images and build a matrix in which each row is an 1D array associated to an image. 
        Otherwise if you have one single image you can scale back it into 2D array. This allows you to calculate n-PC instead of only 1-PC.
    
2. Normalization

    PCA is effected by scale so you need to scale the features in your data before applying PCA
    The normalization consists of mean centering (zero mean) and, possibly, normalizing each variable’s variance to make it equal to 1 (standard deviation equals to 1)
    
    Use StandardScaler or do it manually
    
    ```python
    import numpy as np
    
    mean = np.mean(x)
    std = np.std(x)
 
    x = (x - mean) / std
    ```

3. PC extraction
    
    Extract the *n* principal components from matrix x, fit the model with matrix and apply the dimensionality reduction to the matrix
    
    ```python
    from sklearn.decomposition import PCA
 
    pca2 = PCA(n_components=2)  # For example 2-PC
    x_t_2 = pca2.fit_transform(x)
    ```
    The original data which is n-dimensional it's converted into n_components-dimensions
    If you want you can print the explained variance
    
    ```python
    print(pca2.explained_variance_ratio_.cumsum())
    ```
    The explained variance tells you how much information (variance) can be attributed to each of the principal components. This is important as while you can convert n-dimensional space to n_components-dimensional (with *n > n_components*) space, you lose some of the variance (information) when you do this.

4. PC visualization 

    ```python
    import matplotlib.pyplot as plt
 
    plt.figure('Photo: ' + str(photo_id))
    plt.title('2-PC Components')
    plt.subplot(221)
 
    for i in range(0, ncomp, +2):
        plt.scatter(x[photo_id, i], x[photo_id, i+1], c=color)
    plt.show()
    ```
    The variable *ncomp* is the *n_component* of the PCA used before. 
    
    - The *for* is usefull for high *n_components* PCA. In this example *n_components* is equal to 2 so you can also do
        ```python
        plt.scatter(x[photo_id, 0], x[photo_id, 1], c=color)
        plt.show()
        ```
5. Image reconstruction

    - Choose a photo from the data set
    - Print the images
    
        ##### Print original image
        
        Remember you have to:
        1. De-Standardize matrix
        2. Re-scale the image from 1D (or 2D) into 3D
    
        ```python
        plt.subplot(1, 2, 1)
        x = (x * std) + mean  # De-Standardizing matrix
        original = np.reshape(x[photo_id], (227, 227, 3)).astype(int)  # From 1D to 3D
        plt.imshow(original, interpolation='nearest')
        plt.title('Original', fontsize=20)
        ```
        ##### Print the *n_components*-PC image
        
        Remember you have to:
        1. Do the inverse transform
        2. De-Standardize matrix
        3. Re-scale the image from 1D (or 2D) into 3D

        ```python
        plt.subplot(1, 2, 2)
        approx_2 = pca2.inverse_transform(x_t_2)
        approx_2 = (approx_2 * std) + mean  # De-Standardizing matrix
        img_2_components = np.reshape(approx_2[photo_id], (227, 227, 3)).astype(int)  # From 1D to 3D
        plt.imshow(img_2_components, interpolation='nearest')
        plt.title('2-PC', fontsize=20)
            ```
    - Plot the images

        ```python
        plt.show()
        ```

# PCA application

- One of the most important applications of PCA is for speeding up machine learning algorithms.
- [Quantitative finance](https://en.wikipedia.org/wiki/Principal_component_analysis#Quantitative_finance): principal component analysis can be directly applied to the risk management of interest rate derivatives portfolios.
- [Neuroscience](https://en.wikipedia.org/wiki/Principal_component_analysis#Neuroscience): to identify the specific properties of a stimulus that increase a neuron's probability of generating an action potential.

# Pros and Cons

- Pros
    - Dramatic reduction in size of data
    - Allow estimating probabilities in high dimensional data
        - No need to assume independence
    - Ignore noise

- Cons
    - Can be too expensive for any application
    - Can only capture linear variation

# Utility

- [Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Matplotlib](https://matplotlib.org/users/image_tutorial.html)
- [Matplot colors](https://matplotlib.org/api/colors_api.html)
- [Scatter plot](http://chris35wills.github.io/courses/PythonPackages_matplotlib/matplotlib_scatter/)
- [Numpy ravel](http://pythonforbeginners.com)

# To do

- [ ] Report of different results
- [X] [Example with one single photo](https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com)