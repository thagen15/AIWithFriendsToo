# AIWithFriendsToo
Repository for AI Project 2


Project 2
Group members:
Jason Abel
Thomas Hagen
Parmenion Patias

Github Repo: https://github.com/thagen15/AIWithFriendsToo

Preprocessing  	
 	As it can be seen in our code (file preprocessing.py), we start by loading the MNIST images and labels. We turned the 28x28 images in 784x1 and we put it next to the label data (in one hot encoding) that corresponds to it. We then shuffle the data and split the images from the labels again. We then spit the data in training, valuation and testing sets (60%, 15%, 25% respectively). Once all that is done we are ready to feed the data in the algorithms and test the models.
Neural Networks
Model & Training Procedure Description 
Since this was the first model we made and the first time we tested our data sets and metrics, we started by using the ANN in its simplest form (based on the projects restrictions) and only used the relu and a softmax nodes. That way we were able to see that all the preprocessing was done right and that our metrics and confusion matrix worked well.

	It was now time for us to start testing the effects of epochs and nodes on the model. We first added a tanh (relu, tanh, softmax) node and tested the model with 10, 100, 1000 and 10000 epochs to see the effect of epoch value to the model performance. We then added one more tanh (relu, tanh, tanh, softmax) node to the original model and repeated the 4 epoch values to see if we could find the best performing epoch value range. Finally, we then changed the second tanh with a selu (relu, tanh, selu, softmax) and repeated the 4 epoch values. 
Graph 
Three graphs are displayed below showing the training accuracy and validation accuracy for three different models (RTS (relu, tanh, softmax), RTTS (relu, tanh, tanh, softmax), RTSS (relu, tanh, selu, softmax)). We trained these models with 500 epochs to provide a decent data set for the graphs. As shown in  the graphs, the training accuracy starts below the validation accuracy but over time, the training accuracy eventually overtakes the validation accuracy. It can also be seen that the accuracies start out really low ( <30%) and eventually become really high (>80%).

Visualization 


 This was written as a 2, but the program though it was a 7. It’s a logical error given that even we almost thought that it was a 7. The way it was written makes it really difficult to tell whether its a 2 or a 7.










This was written as a 3 but the program thought that it was a 7 again. The issue again lies in the way the number was written and the very low quality of the MNIST image quality.










This is an 8 that the program thought was a 6. This is an understandable mistake because there is no gap in the top circle of the 8 making it look like a weird 6.





Model Performance & Confusion Matrix
The table below shows the F1 metrics obtained from our testing on 10, 100, 1000, 10000 epochs for each of the three models we tested.

F1 Metrics
Epochs
Relu Tanh Softmax
Relu Tanh Tanh Softmax
Relu Tanh Selu Softmax
10
0.3
0.38
0.44
100
0.75
0.65
0.61
1000
0.77
0.79
0.83
10000
0.86
0.88
0.87
average
0.67
0.68
0.69

We thus see that the number of epochs increases the performance of the ANN. The model that works the best for the highest number of epochs (10000) is Relu-Tanh-Tanh-Softmax, but the model that worked the best overall on average is Relu-Tanh-Selu-Softmax.


















Decision Trees
Feature Extraction & Explanation 
We chose five different methods for feature extraction of the raw pixel data. Some of our different features worked out much better in terms of predicting the number while others were very poor at determining the number.

Average Pixel Density
	Our first method was to get the average pixel density of each picture. This was done with two for loops iterating the the picture list gathering pixel data. Each pictures average pixel intensity number was then calculated. This method of feature extraction did not work out very well and the predicted values were all over the place.
Picture Sorted by Pixel Density
	We iterated through each picture and sorted the pixel list based on intensity values. This helped out quite a bit and was our largest jump in F1 scores. This helps because it gives us a general shape based on where the black and grey pixels start in the list.
Difference of Black Pixels to White Pixels
	We counted the number of black pixels in each picture as well as the number of white pixels and took the difference between the two. This feature didn’t help a ton because there are discrepancies between the size of the handwriting, thus making more black pixels for the same numbers if the handwriting was larger.
Indices Sum
We tried to train the data set based on the indices values of the black spaces. We gathered the sum of all the black pixel indices in the picture and trained the data based of of this sum. Since the positions of the white and black pixels are different for every number, this would hopefully help the program classify the numbers better.
Sum of Pixels
For our last feature, we took the sum of all the pixel values to train the data set. Since some pictures will have more white than others, the number should vary depending on the number in the picture.
	



Description of Experiments 
In the table below, we tested our decision tree with different max depth values to see how the F1 score changed.

Max Depth Value
F1 Score
Baseline
0.77
2
0.07
5
0.57
8
0.78
10
0.79
12
0.78
25
0.76


Through experimentation, we discovered that a max depth of 10 yielded our highest f1 score for the decision tree. This value yielded better values than going through the whole tree (baseline) as well. This could be because it has less chances to determine true positives, true negatives, false positives and false negatives while still classifying correctly. If you go higher than 10, maybe it finds less true positives thus providing a worse F1 score. If you go too small for depth, it may not have enough changes to determine these values to provide the best F1 score.
Model Performance & Confusion Matrix
Below is the baseline model of our decision tree which goes through the whole tree.

Below is the model performance and confusion matrix of our most effective max depth limitation.
















Below is the model performance with all of our feature extractions and depth limiti. In the end the F1 score didn’t improve too much but on average our F1 scores with the features were generally higher than our scores without them.


KNN
Description of Experiments 
To evaluate k values for kNN we tested 8 different values, starting from 1 and going up to 750. 
The results were the following:

K values
Precision
Recall
F1-score
1
0.94
0.94
0.94
3
0.95
0.93
0.94
5
0.96
0.92
0.94
10
0.96
0.89
0.92
50
0.94
0.81
0.87
100
0.95
0.73
0.81
333
0.84
0.41
0.49
750
0.09
0.06
0.07

There are a couple of inferences we can make by looking at the precision, recall and f1-score metrics of different k-values. First of all we can see that while both precision and recall values decrease as we increase the k-value, recall decreases at a faster rate than precision. For example at k=100 precision is at 95% but recall has already dropped down to 73%. This is even more apparent when looking at the metrics when k=333. However there seems to be a value after which precision will decrease much faster than recall since at k=750 precision seems to have caught up with recall and the values are 9% and 6% respectively. 

When looking at f1-score (often considered a better metric as it incorporates both precision and recall in it) the metric’s value is the same when k=1, k=3 and k=5. That is because precision increases between the values k=1 and k=10 and then it drops, while recall seems to be decreasing from k=1 onwards.
Model Performance & Confusion Matrix 
For k=3 (we decided to use this value as it has the second highest precision and recall value that we managed to score (95% and 93% respectively) and the highest f1-score values (94%).


The model seems to have a problem identifying the number 0 correctly as we can see in the confusion matrix that is missed the chance to do so quite some times. No such problems exist in the rest of the numbers.
Visualization

In this example the kNN seemed to mix 2 with 8. This seems like a logical mistake as the 2 provided is written in an unorthodox way and is really close to an 8.
















In this case the program mistook a 7 to be a 0. Our team initially took the 7 to be a 9, so we understand how the kNN 















In this example see that the kNN mistook the 2 to be a 3. Given that the 2 looks more than an upside e than anything else it is logical that the program mistook the number as well.














All in all we see that when the kNN mistakes a value it compares it with 3 images of same value. Which means that these 3 same value numbers are the closest to the mage tested.
