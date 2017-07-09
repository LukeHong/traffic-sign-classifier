# **Traffic Sign Recognition**

A project from Udacity Self-Driving Car NanoDegree to recognize traffic sign from Germany Traffic Sign Dataset.

---

### **Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/all_class.jpg "Image For Each Class"
[image2]: ./examples/samples.jpg "Sample Counts"
[image3]: ./examples/normalization.jpg "normalization"
[image4]: ./new_images/7.jpg "Traffic Sign 1"
[image5]: ./new_images/12.jpg "Traffic Sign 2"
[image6]: ./new_images/18.jpg "Traffic Sign 3"
[image7]: ./new_images/33.jpg "Traffic Sign 4"
[image8]: ./new_images/35.jpg "Traffic Sign 5"
[image9]: ./new_images/38.jpg "Traffic Sign 6"
[image10]: ./examples/predict.jpg "General Caution Predict"

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I have calculated summary statistics of the traffic signs data set:

* The size of training set is `34799`
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `32x32x3 (RGB)`
* The number of unique classes/labels in the data set is `43`

#### 2. Include an exploratory visualization of the dataset.

First, I display a image from all classes in training set to see what each class is.

![Each Class][image1]

Secound, I made a bar chart of how many samples in every classes that shows different number of samples for each class.
![Sample Counts][image2]



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques?

As a first step, I normalized the image data to get a better performance.

![Normalization][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x6	  			    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16	  			    |
| flatten				| outputs 400                                   |
| Fully connected		| outputs 120  									|
| Fully connected		| outputs 84  									|
| Fully connected		| outputs 43  									|
| Softmax				| outputs 43           									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used:
* The batch size is `128`
* The optimizer is `Adam`
* The learning rate is `0.0017`

Furthermore, I made `Early Stop` and `Best Model` mechanism

**Early Stop**

I use early stop to avoid overfitting.
I set a **early_stop_threshold**, **early_stop_count**, and **previous_val_acc**.
After each epoch I will check whether the validation accuracy is better than previous epoch or not. If the validation accuracy is worse than previous one for **early_stop_threshold** times continuously, then the training process will be stop.

The final **early_stop_threshold** is **5**.

**Best Model**

I set a **best_accuracy** to store the best number of validation accuracy. After each epoch, the current model will be saved as **best model** if the validation accuracy is better than the **best_accuracy**.

I will get the model of best validation accuracy in the whole training process, instead of the last model in training process.



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of **0.956**
* test set accuracy of **0.937**

I choose the LeNet architecture to find how best it could be.
After some tuning then I found some problem. If I set **low epoch** number then I might not get a better model as result. But if I set **high epoch** number then I might get a overfitting model.

To solve these problems, I made early stop to avoid overfitting at first. I won't get a overfitting model after that, but there's one more problem appears.
it might trigger the early stop mechanism **too early** that model is not trained enough, so I increase the **early_stop_threshold**.

After I used the early stop, I still can't get the model of best validation accuracy.
I used a variable to store the best accuracy number, then I can keep the best model in training process.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![100 km/h][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8] ![alt text][image9]

The image of general caution sign might be difficult to classify because
the upper part of the sign is unclearly.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Priority Road   		| Priority Road									|
| General Caution		| **Pedestrians**    							|
| Turn Right Ahead		| Turn Right Ahead								|
| 100 km/h	      		| Bumpy Road					 				|
| Ahead Only			| Ahead Only         							|
| Keep Right			| Keep Right         							|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.0%. This compares favorably to the accuracy on the test set of 93%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![Predict][image10]

The model predict most of these images clearly for almost 100%. Because the image of general caution sign is not so clear. So in the result of general caution sign, the top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .65         			| Pedestrians  									|
| .35     				| General Caution								|
