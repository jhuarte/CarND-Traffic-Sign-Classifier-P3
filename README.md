# **Project 3: Traffic Sign Recognition** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview

On this project, I implemented a convolutional neural network (CNN's) based on the [LeNet](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) architecture developed by [Yann Lecun](http://yann.lecun.com/) and implement traffic signs recognition system. The dataset used to train, validate and test the CNN are the images from the German Traffic Sign Dataset.

There are three importants files that explain and document the project:

* the Ipython notebook with the code
* the code exported as an html file
* a writeup report (this document) 
---

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/visualization.png "Dataset distribution (train (B), validation (G), test (O))"
[image2]: ./images/signs_representation.png "Dataset representation"
[image3]: ./images/ext_60kmh.png "Preprocessing images"
[image4]: ./images/descarga.png "Web images"
[image5]: ./images/tv_1_accuracy.png
[image6]: ./images/tv_2_accuracy.png
[image7]: ./images/p1.png "Web images"
[image8]: ./images/p2.png
[image9]: ./images/p3.png

## Dependencies
In order to fullfil all the specications of the project, I decided to use Google Colab. The idea is to speed up as much as possible the training process using the GPU/TPU functionality. 

![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)

A I've stored the dataset on google drive space, the first thing I did was to grant access to the working directory. For that purpose I create a variable named `root_dir` to point to my data workspace. If you want to run this code on your own system remember to update this variable.

```python
from google.colab import drive
drive.mount('/content/drive')
```
```python
root_dir = '/content/drive/MyDrive/Colab Notebooks/CarND-Traffic-Sign-Classifier-Project/'
```

As default, colab use the version 2 of `tensorflow` at this moment, so all the code learned from the leassons have been updated to this version (including `keras`functions). 

```python 
# Check the tensor flow version
print(tf.__version__)
2.4.1
```

## Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the `pandas`,`numpy` and `matplotlib.pyplot` libraroes to calculate summary statistics of the traffic signs data set and plot the results. The general numbers obtained are:

* Number of training examples = `34799`
* Number of testing examples = `12630`
* Image data shape = `(32, 32)`
* Number of classes = `43`

I've used `pandas` to read the `cvs` file and obtain the description of each `sign` (`signals = pd.read_csv(root_dir + 'signnames.csv')`. 

```python
# Read the CVS file with the ClassId's and the signal names
signals = pd.read_csv(root_dir + 'signnames.csv')

def plt_bar_dataset(y_train, y_test, y_valid):
  # Explore the training dataset
  train_values, train_counts = np.unique(y_train, return_counts = True)
  plt.bar(train_values, train_counts)
  # Explore the test dataset
  test_values, test_counts = np.unique(y_test, return_counts = True)
  plt.bar(test_values, test_counts)
  # Explore the test dataset
  valid_values, valid_counts = np.unique(y_valid, return_counts = True)
  plt.bar(valid_values, valid_counts)

  return train_counts, test_counts, valid_counts

#Plor the original distribution of the data set
train_counts = plt_bar_dataset(y_train, y_test, y_valid)
```

In the next represent the distribution of the diferent signs on the `train dataset (blue)', 'validation dataset (orange)` and `test dataset (green)`. As we can see the distribution is not uniform and we have some `classes (signs)` with a very short ocurrence (ie: `sign_class = 0 (Speed limit (20km/h))` and other with very high ocurrence (ie: `sign_class = 1 (Speed limit (50km/h))`.

![alt text][image1]

#### 2. Include an exploratory visualization of the dataset.

The next step is to analyze the images (color, blurring, etc.). The code used to represent only the first image of each class on the dataset is the next one.

```python
# Represent the first ocurrence on the train dataser of each classes (signals)

def plot_images(images, labels):  
  fig, axes = plt.subplots(5, 10 , figsize = (15 , 10))
  fig.subplots_adjust(hspace = 0.2, wspace = 0.1)
    
  for idx, ax in enumerate(axes.flat):
        if idx < n_classes:
          ax.imshow(images[idx])
          xlabel = "{0}".format(idx)
          ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])    
  plt.show()

# Obtain the index of the first ocurrence for each signal
idxes = []
for sign in range(n_classes):
  idx = (np.where(y_train == sign)[0][0])
  idxes.append(idx)        

# Plot one image for each class and the label
plot_images(x_train[idxes], signals['SignName'])
```

![alt text][image2]

As general onclusions:
* The signs (classes) in the data set are not well balanced (not a lot of training examples for all classes)
* Different quality of the images (some of them are very difficult for humans also)
* Very extreme lighting conditions (night, day)

As we will see later some preprocesing or data augmentation of the images will be needed to train sucessfully the CNN. The approach expoused in [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) will be a good approach. 

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

So, let's start with the fun part!. As we mention on the previous point some preprocessing should be done as suggest [Yann Lecun](http://yann.lecun.com/) in [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). A `preprocessing(images)` function has been designed that:

```python
def preprocessing(images):    
  """
  Pre-process pipeline: grayscale transformation, equilization and normalization
  
  Parameters:
    images (numpy.ndarray): (dim(images), 32, 32, 3)
  Returns:
    batch (numpy.ndarray): (dim(images), 32, 32, 1)
  """      
  shape = images.shape 
  out_img_shape=(shape[1],shape[2],1)
  batch = np.zeros((shape[0],shape[1],shape[2],1))
    
  for i in range(len(images)):
    gray = apply_grayscale(images[i,:]  )        
    clahe = apply_clahe(gray).reshape(out_img_shape)
    norm = apply_normalize(clahe)        
    batch[i] = norm
  
  return batch
  ```


* Converts images to grayscale using `OpenCV` function.

```python
def apply_grayscale(img):
  """
  Apply grayscale transformation using openCV function (cvtColor)
    
  Parameters:
    img (numpy.ndarray): Color image

  Returns:
    img (numpy.ndarray): Grayscaled image    
  """
  shape = img.shape
  img = np.array(img, dtype = np.uint8)    
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray = gray[: , :, np.newaxis]
  return gray
```

* Apply `CLAHE algorithm` to equilization 

```python
def apply_clahe(img):
  """
  Apply CLAHE (Adaptive histogram equalization) to the image (img)
    
  Parameters:
    img (numpy.ndarray): Grayscaled image (unint8)

  Returns:
    img (numpy.ndarray): Equalizated image    
  """    
  clahe = cv2.createCLAHE(clipLimit = 4.0, tileGridSize = (4, 4))
  clahe_img = clahe.apply(img)
  return clahe_img
  ```
  
* And perform a basic normalization image to a range of [-1:1]

```python
def apply_normalize(img):
  """
  Normalize the image to (-1, 1) range. (other methods can be used like mean/std ...etc.)
  
  Parameters:
    img (numpy.ndarray): Grayscaled image (unint8)
  Returns:
    img (numpy.ndarray): Normalized image    
  """   
  return (img - 128.) / 128.
```

Here are some examples of a traffic signs images before and after preprocessing (this images correspond to the new images used for the final section).

![alt text][image3]

I decided to generate additional data because as we can see on the training dataset histogram, the number of signals are not well balance and this would create some bias on the prdections done by the CNN. To add more data to the the data set, I used the following techniques because I tried to balance the number of class signs over the dataset. Basically the idea is to fix a minimun number of images for each classes and using the `ImageDataGenerator` function create new images. For simplicity I avoided to flip the images and keep the numbers and letters on the right direction (smarter augmentation could be done).

```python
def augment_dataset(x_dataset, y_dataset, min_occurrence):
  # Configure the image generator to create rescaled, shifted, rotated or zomm images (not flipped -> numbers and letters)
  train_gen = tf.keras.preprocessing.image.ImageDataGenerator(height_shift_range = random.uniform(-0.4, 0.4),
                                                              rotation_range = random.uniform(-15., 15.), 
                                                              zoom_range = [random.uniform(0.7, 0.7),random.uniform(0.9, 0.9)], dtype = np.uint8 )

  # For each class (sign) check the number of ocurrences and complete as needed
  for sign_class in range(n_classes):
    # Prepare x for the first iteration
    x = x_dataset[y_dataset == sign_class]
    # Init for each class
    x_dataset_augmented = np.zeros([1,32,32,3])
    y_dataset_augmented = np.zeros([1])

    while ((min_occurrence - len(x)) > 0):
      it = train_gen.flow(x = x, y = None,
                          batch_size = min_occurrence - len(x),
                          shuffle = True)
      batch = it.next()
      # Update x to the next iteration
      x = np.concatenate((x, x_dataset_augmented), axis = 0)
      # Store augmented data 
      x_dataset_augmented = np.concatenate((x_dataset_augmented, batch), axis = 0)
      y_dataset_augmented = np.concatenate((y_dataset_augmented, np.full(batch.shape[0], sign_class)))      

    # Update the data set
    x_dataset = np.concatenate((x_dataset, x_dataset_augmented), axis = 0)
    y_dataset = np.concatenate((y_dataset, y_dataset_augmented), axis = 0)

  return x_dataset, y_dataset
 ```
 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers, based on the LeNet architecture:

| Layer (type)         		|     Output Shape	        					| Params | Parameters|
|:---------------------:|:---------------------------------------------:| :---------------------:|:---------------------:|
| Input         		| 32x32x1 Grayscale   							| ||
| conv2d (Conv2D)     	| (None, 28, 28, 6)  	| 156|`conv1_filters, (5, 5), activation = 'relu'`|
| max_pooling2d (MaxPooling2D)					|	(None, 14, 14, 6)											| 0|`pool_size = (2, 2), strides = 2, padding = 'SAME'`|
| conv2d (Conv2D)     	| (None, 10, 10, 16)  	| 2416|`conv2_filters, (5, 5), activation = 'relu'`|
| max_pooling2d (MaxPooling2D)					|	(None, 5, 5, 16)											| 0|`pool_size = (2, 2), strides = 2, padding = 'SAME'`|
| flatten (Flatten)					|	(None, 400)											|0||
| dense (Dense)					|	(None, 120)											|48120||
| dropout (Dropout)					|	(None, 120)											|0|`rate = 0.5`|
| dense_1 (Dense)					|	(None, 84)											|10164|`activation = 'relu'`|
| dropout_1 (Dropout)					|	(None, 84)											|0|`rate = 0.5`|
| dense_2 (Dense)|(None, 43)|       |3655|`activation = 'relu`|

|  | Params|
|:------------:|:------------------------:|
| Total params:| 64,511|
| Trainable params:| 64,511|
| Non-trainable params:| 0|

```python
# Parameters
conv1_filters = 6
conv2_filters = 16
fc1_units = 120
fc2_units = 84
num_classes = n_classes

# Architecture
model = models.Sequential()
# Layers
# Convolutional layers
model.add(layers.Conv2D(conv1_filters, (5, 5), activation = 'relu', input_shape = (32, 32, 1)))
model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'SAME'))
model.add(layers.Conv2D(conv2_filters, (5, 5), activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'SAME'))
# Fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(fc1_units, activation = 'relu'))
model.add(layers.Dropout(rate = 0.5))
model.add(layers.Dense(fc2_units, activation = 'relu'))
model.add(layers.Dropout(rate = 0.5))

model.add(layers.Dense(num_classes))
```

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, at the begining I used the stotastic gradient descent optimizer but after some trials a research I decide to change to ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) the `Adam` optimizer, that is in general a better optimizaer in terms of performance. 

```python
pt = optimizers.Adam(learning_rate = 0.001)

model.compile(optimizer = opt,
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])
```

I tunned four different hyperparamters to validate the system:

* Learning rate: `0.001`
* Batch size: `128`
* Number of epeochs: `40`
* Steps per epoch: `math.ceil(len(x_train) / batch_size)`

I tried different values of each hyperparameters, specially for the `learning_rate` and the number of `epochs` until I decided to change all the proyecto to colab. batch sizes `(32, 64 and 128)` and `128` was a good balance between speed and accuracy. As we can see on the section, from epoch `10` the accuracy is more os less stable.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I started with the LeNet architecture witout any type of regularization and the system prepared for RGB images `(32,32,32,3)`. I trainned the CNN with the training dataset (without any preprocessing) and the results was not very good from what was expected, so clearly there was a problem of under fitting.

![alt text][image5]

```
Epoch 40/40
1088/1088 [==============================] - 3s 3ms/step - loss: 0.4410 - accuracy: 0.8919 - val_loss: 0.4222 - val_accuracy: 0.8973
```

Then I implemented the preprocessing functions and I've changed a little bit the architecture changing the input layer to `(32,32,32,1)` (grayscale). At the sametime and with the idea to avoid the overfiting of the CNN I modified the solution adding `dropout` regularization on the final layers (the fully connected layers). I also add some new images to the dataset with the augmentation function and train again the CNN. 

My final model results were:
* training set accuracy of 0.9757
* validation set accuracy of 0.9803
* test set accuracy of 0.9617


![alt text][image6]
```
Epoch 40/40
1088/1088 [==============================] - 3s 3ms/step - loss: 0.0815 - accuracy: 0.9757 - val_loss: 0.0848 - val_accuracy: 0.9803

395/395 [==============================] - 1s 2ms/step - loss: 0.1916 - accuracy: 0.9618
Accuracy on test dataset: 0.9617577195167542
```

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4]

The second image (the original) might be difficult to classify because the orientation of the worker in on the opposite direction compared with the data set. Then I've two options. The first one is to create a new dataset of images for this sign or flip manually the image to be coherent with the dataset. I choosed the second one just for simplicity.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction using the `p = model.predict(images)`:

![alt text][image3]

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Road work    			| General caution 										|
| Yield					| Yield											|
| 60 km/h	      		| 60 km/h						 				|
| Turn right ahead			| Turn right ahead      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80% wich is worst than the accuracy measured with the testdataset. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

To calculate the softmax probabilities I used the `values, indices = tf.nn.top_k(tf.nn.softmax(p),5, sorted=True)` that return the probabilities for each prediction 

For the first image, the model is predict a stop sign (probability of 0.32) but it's a very low value.

| Probability			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.32     		| Speed limit (60km/h)   									| 
| 0.28    			| Double curve										|
| 0.30					| Speed limit (80km/h)											|
| 0.07	      		| Keep right						 				|
| 0.03			| Bicycles crossing      							|

![alt text][image7]

For the second image the prediction is much better, 0.92 of probability of been a `Stop`.

| Probability			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.92     		| Stop   									| 
| 0.04    			| Speed limit (60km/h)										|
| 0.02					| Go straight or right											|
| 0.001	      		| Turn left ahead						 				|
| 0.0004			| Ahead only      							|

![alt text][image8]

For the third the prediction is very good, the system is really sure that the sign is a `Yield`:

| Probability			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00     		| Yield   									| 
| 0.00    			| Ahead only										|
| 0.00					| Priority road											|
| 0.00	      		| Turn left ahead					 				|
| 0.00			| Speed limit (20km/h)     							|

![alt text][image9]

#### Conclusions

I've learn a lot with the project, indeed was my first experience with neuronal networks. I know that the system is not perfect and I could have made many improvements like better augmentation (smarter strategy), L2 regularization but 96% of accuracy for the test set is avery good result for the first experience. 


#### Notes
* The visualization of the architecture graph can be enlightening. [TensorBoard](https://www.tensorflow.org/tensorboard) is a very useful tool for that.
* Here is a discussion on how to choose the [batch size](https://stats.stackexchange.com/questions/140811/how-large-should-the-batch-size-be-for-stochastic-gradient-descent) of Stochastic Gradient Descent.
* Here is a discussion on the [Adam Optimizer](https://ruder.io/optimizing-gradient-descent/index.html#adam).
* For hyperparameter optimization, you can refer to this [source](https://cs231n.github.io/neural-networks-3/#hyper).

Things to consider at the time of designing a CNN:

* Data preprocessing
* Number and type of layers
* Selection of the adequate optimizer
* Tunning of hyperparameters
* Model training
* Model's assessment metric/benchmark

The plot of the learning curve helps to see if there is a considerable gap between the training and validation accuracy. If there is one, the model may be overfitting or underfitting. More information about the learning curve here.
The following regularization techniques can be used to minimize overfitting to training data(in addition to dropout and cross-validation that you already use):

* [L2 Regularization](https://keras.rstudio.com/reference/regularizer_l1.html). Using lambda = 0.0001 which seemed to perform best. The important point here is that L2 loss should only include weights of the fully connected layers, and normally it doesn’t include bias term. The intuition behind it is that bias term is not contributing to overfitting, as it is not adding any new degree of freedom to a model. Here is [An Overview of Regularization Techniques in Deep Learning](https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/).

* [Early stopping](https://deeplearning4j.konduit.ai/tuning-and-training/early-stopping).You can apply stopping with the patience of 100 epochs to capture the last best-performing weights and roll back when the model starts overfitting training data. A validation set [cross entropy loss](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html) as an early stopping metric, the intuition behind using it instead of accuracy is that if your model is confident about its predictions it should generalize better. Please check this article to [know How to Stop Training Deep Neural Networks At the Right Time Using Early Stopping](https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/).

* [Batch Normalization](https://towardsdatascience.com/batch-normalization-theory-and-how-to-use-it-with-tensorflow-1892ca0173ad) is not a regularization technique, but it helps neural network weights to convert faster and hence works well with early stopping.

