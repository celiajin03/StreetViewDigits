# CNN

➢ Detailed explanation of how CNN works.

Traditional neural networks are comprised of node layers, containing an input layer, one or more hidden layers, and an output layer. Each node connects to another and has an associated weight and threshold. Convolutional neural networks (CNN) are distinguished from the traditional ones by their superior performance with image, speech, or audio signal inputs. 

A CNN typically consists of three components:

+ **Convolutional layers**: apply many convolution filters to the image. Then usually apply an activation function (e.g., Relu) to the filtered image to make the model non-linear. Output of convolutional layer is usually called a feature/activation map.

<img src="https://miro.medium.com/max/1838/1*xBkRA7cVyXGHIrtngV3qlg.png" alt="Convolutional Neural Networks — A Beginner&#39;s Guide | by Krut Patel |  Towards Data Science" style="zoom: 33%;" />

​		Concolution Operation
​		source: [towardsdatascience](https://towardsdatascience.com/convolution-neural-networks-a-beginners-guide-implementing-a-mnist-hand-written-digit-8aa60330d022)

<img src="https://miro.medium.com/max/652/1*NsiYxt8tPDQyjyH3C08PVA@2x.png" alt="img" style="zoom:50%;" />

​														Movement of the Kernel
​														source: [towardsdatascience](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

> The convolution operation is basically taking the dot product of the image and the kernel. The kernel will traverse across the whole image in the way illustrated above with a specified stride value. Applying the filter in this way allows the CNN model to extract features from the images.

+ **Pooling (subsampling) layers**: down sample the feature map to reduce its dimensionality in order to decrease processing time, e.g., 2x2 max pooling keeps their maximum value of 2x2 sub-regions of the feature map and discards all other values.

<img src="https://miro.medium.com/max/1000/1*KQIEqhxzICU7thjaQBfPBQ.png" alt="img" style="zoom:67%;" />

​									Two Types of Pooling
​									source: [towardsdatascience](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

> The graph shows an example each for max pooling and average pooling. We can see that it's basically a way of downward sampling, as we only take a single value from the original 2×2 region. This reduces the input dimension and expedites the model training process.

+ **Dense (fully connected) layers**: perform classification on the features extracted by the convolutional layers and down-sampled by the pooling layers, where each node in the dense layer is connected to every node in the preceding layer.

<img src="https://miro.medium.com/max/850/1*GLQjM9k0gZ14nYF0XmkRWQ.png" alt="img" style="zoom:67%;" />

​										Flattening of a 3x3 image matrix into a 9x1 vector
​										source: [towardsdatascience](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

> The fully connected layers are similar to the hidden layers in tranditional neural networks. We usually first perform a flattening operation on the resulting input from previous steps (as shown in the graph above). This step flattens the input but does not affect the batch size. Then we will make the model prediction based on the vector input using these dense layers.



<img src="https://miro.medium.com/max/1400/1*uAeANQIOQPqWZnnuH-VEyw.jpeg" alt="img" style="zoom: 50%;" />

​	A CNN sequence to classify handwritten digits
​	Input - CONV - POOL - FC
​	source: [towardsdatascience](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

To summarize, a CNN model basically chained all the aforementioned layers together. And by using different layer combinations, different parameters or optimizing methods, we aim to achieve the most satisfying prediction accuracy by extracting the valuable features from the original input. These high number of filters essentially learn to capture spatial features from the image based on the learned weights through back propagation. Hence they can successfully boil down a given image into a highly abstracted representation which is easy for predicting.



# Parameters

➢ Explanation of each and every parameter/hyperparameter of CNN with its function.

### Input

**Arguments**

- **shape**: A shape tuple (integers), not including the batch size. For instance, `shape=(32,)` indicates that the expected input will be batches of 32-dimensional vectors. Elements of this tuple can be None; 'None' elements represent dimensions where the shape is not known.
- **batch_size**: optional static batch size (integer).
- **dtype**: The data type expected by the input, as a string (`float32`, `float64`, `int32`...)



### Conv2D

**Arguments**

- **filters**: the number of output filters in the convolution.

> The Number of Channels is the equal to the number of color channels for the input but in later stages is equal to the number of filters we use for the convolution operation. 

- **kernel_size**: n×n, specifying the height and width of the 2D convolution window. In general we use filters with odd sizes.
- **strides**: an integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Stride is the number of pixels to skip while traversing the input horizontally and vertically during convolution.
- **padding**: one of `"valid"` or `"same"`. `"valid"` means no padding. `"same"` results in padding with zeros evenly to the left/right or up/down of the input. Padding is generally used to add columns and rows of zeroes to keep the spatial sizes constant after convolution. When `padding="same"` and `strides=1`, the output will have the same size as the input.
- **activation**: Activation function to use. Choices include `relu`, `sigmoid`, `tanh` and etc. Activation functions are used to make the model non-linear.
- **use_bias**: Boolean, whether the layer uses a bias vector.



### MaxPooling2D

**Arguments**

- **pool_size**: integer or tuple of 2 integers, window size over which to take the maximum. `(2, 2)` will take the max value over a 2x2 pooling window. If only one integer is specified, the same window length will be used for both dimensions.
- **strides**: Specifies how far the pooling window moves for each pooling step. 
- **padding**: One of `"valid"` or `"same"`. `"valid"` means no padding. `"same"` results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.



### BatchNormalization

**Arguments**

- **axis**: Integer, the axis that should be normalized (typically the features axis). For instance, after a `Conv2D` layer with `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.



### Dropout

**Arguments**

- **rate**: Float between 0 and 1. Fraction of the input units to be dropped.

# Code

➢ Detailed explanation of your code logic.

1. SET SEED
    I first set seed to ensure reproducible results for the model.

2. READ DATA
    Read train and test data from .mat files.

3. PREPROCESS
    For X, I move the last axes of array X to be the first in accordance with the input shape of (batch_size, imageside1, imageside2, channels). I scale the images to the [0, 1] range by dividing 255 since it can somehow normalize the data. 
    For y, I convert the class vectors to binary class matrices using one hot encoding.

4. MODEL BUILDING

    I specify the model structure in this part. The `Sequential` class groups a linear stack of layers into a `tf.keras.Model`.

5. TRAINING
    The specified model is tained on training data using a particular batch size and epoch value. I use tensorboard to record the training logs and detect possibilities of under or over fittting.

6. EVALUATION
    The scores are evaluated on the test data to examine the model performance and its ability to make generalizations. Confusion matrix is plotted to provide information on the prediciton of each class.

    



# Starting Model

 ➢ Mention your starting model and how you choose that.

```python
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
```

```python
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0         
_________________________________________________________________
dropout (Dropout)            (None, 1600)              0         
_________________________________________________________________
dense (Dense)                (None, 10)                16010     
=================================================================
Total params: 34,826
Trainable params: 34,826
Non-trainable params: 0
_________________________________________________________________
```

I decided to use the example model (a simple MNIST convnet) provided by Keras documentation site (and this assignment) as the starting point. [https://github.com/keras-team/keras-io/blob/master/examples/vision/mnist_convnet.py] As indicated on the Keras official website, this simple convnet achieves ~99% test accuracy on MNIST, which is fairly decent. Since the MNIST digit recognition task is similar in nature with the one we want to perform for Street View House Numbers (SVHN), I thought this could be a good starting point.

# Final Model

➢ Explain how you reached final model from starting model with intermediate stages as well (including visualization).

**Starting Model:**

<img src="/Users/Celia/Library/Application Support/typora-user-images/截屏2021-12-21 下午4.35.30.png" alt="截屏2021-12-21 下午4.35.30" style="zoom:33%;" /><img src="/Users/Celia/Library/Application Support/typora-user-images/截屏2021-12-21 下午4.37.45.png" alt="截屏2021-12-21 下午4.37.45" style="zoom: 50%;" />

We can see that by deploying the baseline model from the MNIST example, I already achieved a quite decent test accuracy of ~90%. This suggests a quite promising direction, so I decided to only make small modifications based off of that.

Compared with the MNIST digit recognition, our task on Street View House Numbers (SVHN) is certainly more challengeing as the input data is not very clean by itslef. Considering this increase in the level of complexity, I decided to modify the intial model towards a more complex direction as well.

Several ways of achieving that include: increasing the number of filters, adding more convolutional layers. I also tested out with others parameters such as dropout rate and kernel size. 

+ For convolutional layers, I mainly tried 2, 3, 4 layers and decided to keep it at 3. 
+ For number of filters, I tried values of 32, 64, 128 for different combinations (eg, for a three layer it can be 32, 64, 64 / 64, 64, 64 / ... ).
+ For dropout rate, I tried values of 0.4, 0.5 and 0.6.
+ And for kernel size, I mainly tried using only (3×3) or only (5×5) .

Inorder to make the model less delicate to hyperparameter tuning and also shrink the internal covariant shift, I decided to also apply batch normalization to maintain the distribution. After several trials on different combinations, my final model looks something like this:

**Final Model:**

```python
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
```



```python
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 32, 32, 64)        1792      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 16, 16, 64)       0         
 )                                                               
                                                                 
 batch_normalization (BatchN  (None, 16, 16, 64)       256       
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 16, 16, 64)        36928     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 8, 8, 64)         0         
 2D)                                                             
                                                                 
 batch_normalization_1 (Batc  (None, 8, 8, 64)         256       
 hNormalization)                                                 
                                                                 
 conv2d_2 (Conv2D)           (None, 8, 8, 64)          36928     
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 4, 4, 64)         0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 1024)              0         
                                                                 
 dropout (Dropout)           (None, 1024)              0         
                                                                 
 dense (Dense)               (None, 10)                10250     
                                                                 
=================================================================
Total params: 86,410
Trainable params: 86,154
Non-trainable params: 256
_________________________________________________________________
```

# Model evaluation

As shown in the graph, my final model achieved a test accuracy of 92.1%. As we can tell from the loss curve, both the training and validation loss go through a period of decrease and then gradaully plateaued. There's no sign of overfitting as there is no 'U' shape in either the validation accuracy or loss curve. The gap between the training and validation accuracy is relatively small. The only issue is that the value of loss seem to be slightly large in magnitude, which suggests further room for improvements.

![截屏2021-12-21 下午8.50.38](/Users/Celia/Library/Application Support/typora-user-images/截屏2021-12-21 下午8.50.38.png)

<img src="/Users/Celia/Library/Application Support/typora-user-images/截屏2021-12-21 下午9.41.01.png" alt="截屏2021-12-21 下午8.50.38" style="zoom: 33%;" /><img src="/Users/Celia/Library/Application Support/typora-user-images/截屏2021-12-21 下午7.30.53.png" alt="截屏2021-12-21 下午7.30.53" style="zoom:50%;" />

I also plotted the confusion matrix for the test data set. As we can tell from the diagonal, the model did a pretty good job in correctly predicting the numbers. We will get a high precision and recall in this case.

![image-20211221205504329](/Users/Celia/Library/Application Support/typora-user-images/image-20211221205504329.png)

# Difficulties

➢ What are the difficulties faced while implementing the model?

1. Long Computation Time

    The data set itself is quite large and since I usually need at least tens of epochs for training, the model takes quite a bit of time to train. While running on my local device, it uses CPU computation and is very slow to finish one round of training. So I address this by moving towards using Colab and the GPU resources available. However, the free computational resources is still limited and I'm not allowed to continuously use it due to the time and RAM restrictions.

2. Overwhelming Parameter Choices

    There are way too many parameters to be considered in this CNN model, as well as countless possibilities of model structures to try out. Since there isn't a single Golden Rule guideline on parameter choosing, I have to follow the trial-and-error approach to test out reasonable combinationa based on empirical knowledge. The number of combinations can grow factorially, which greatly increase the workload.

3. Randomness

    There are plenty of randomness involved in the whole model training process. Each time the initialization of parameters and weights could be different which means we would get fluctuating results even based on the same setting. Although I tried to address this issue by setting the seeds, it could still mean that the accuracy we get for a particular sturcture is not representative of its true performance. So this made it quite hard to compare between models, especially when the difference in accuracy is not that large. And setting seeds also seem not to work well in Colab.

4. Stop Timing

    In order to avoid under or over fitting, we have to oversee the model training process and it is often a problem to decide when to stop. I have to test out different epoch values and observe how the training and validation loss curve fold out during the process to see if additional epochs are necessary. This is also more of an subjective criterion.

# Improvements

➢ How can you improve the model further?

1. Tune Parameters Further

Due to the time and resource limitation, I certainly can not try out all possible combinations of the parameters. There are a number of parameters that I haven't palyed around with yet, such as using varying kernel size between layers, the method for padding, the choice of activation function, the learning rate, the model optimizer and etc. By tuning these parameters to a better value, it's highly likely that we can get improved model performance. And if resources permitted, we could do a comprehensive GridSearch on the cobinations of these choices and silmutaneously observe their interaction effect.

2. Conduct Data Augmentation

We can generate more image samples using image augmentation techniques, such as rotation, shear, flip and etc. The increase in sample size might contribute to a better model accuracy.

3. Test More Model Structures

Again due to the time and resource limitation, I only worked on a selection of model structures in this task. There are certainly more structures that would potentially work better than the one I adopt currently, especially considering the tons of diffrent types of layer available to be incoporated. We know that a wide neural network is good at memorization, while a deep one could be better at generalization. So this might be a potential direction of improvement to make.

4. Address Imbalanceness

As we can see from the count of labels, this Street View House Numbers (SVHN) data set is not balnaced in terms of the digits included. `1` has the highest appearance of 5099 cases compared to `9` with the lowest frequency of only 1595 cases.  This imbalanceness may affect the model performance and we can deal with it using a variety of under or over sampling methods. Another approach would be to associate each lable with different weight in order to account for the fact of imbalanceness.





## References

+ <5293 Class Notes>
+ https://keras.io/api/layers/
+ https://github.com/keras-team/keras-io/blob/master/examples/vision/mnist_convnet.py
+ https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/  
+ https://www.ibm.com/cloud/learn/convolutional-neural-networks
+ https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7
+ https://towardsdatascience.com/convolution-neural-networks-a-beginners-guide-implementing-a-mnist-hand-written-digit-8aa60330d022





