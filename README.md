# Perceptron-Algorithm
The problem of parsing a set of 50 points in a 6-dimensional plane without using Neural Networks libraries.
## Perceptron Algorithm 
The perceptron model is more general than the McCulloch-Pitts neuron in terms of computation. It takes an input, aggregates it (weighted sum), and only returns 1 if the aggregated amount above a certain threshold; otherwise, it returns 0.We'd end up with something like this if we rewrote the threshold as indicated above as a constant input with a variable weight:    <br/>
![perceptron](https://user-images.githubusercontent.com/78887209/157613277-2ccb85ef-5ab9-4d4e-b411-f2de1655090c.png) <br/>
Only linearly separable functions can be implemented with a single perceptron. It accepts both real and boolean inputs and assigns a set of weights and a bias to each.  <br\>
In the first question, we first determined 50 points between -0.5 and 0.5 on the 6-dimensional plane.
We placed it in randomlist[]. Then we defined the class as -1 if the 3rd Number in the randomlist is greater than zero, and class as 1 if it is less. Thus, we can collect 50 points on the 6-dimensional plane into one. We can correctly divide it into two classes. <br/>
We mixed our data first and then assigned our first 30 data to the training set and the remaining 20 data to our test set. <br/>
First we trained and tested our data, since the dataset is linearly separable, our accuracy rate is 1. <br/>
Afterwards, we wanted to change some parameters and look at the effect on the training and testing process. <br/>
### a.Initial conditions of weights
In many different tests we did, we did not observe a change in the success rate and the time it took to reach the correct weights when we chose the initial values ​​of the weights too small or high. On the other hand, when we increase the initial values ​​of the weights too much, the success rate decreases and we observed a significant increase in the number of steps to reach the correct weights.
### b.Learning Rate
We did not observe a change when we increased the learning rate.When we reduce the learning rate below 0.1, there is no obvious change in our success rate.
However, the number of steps required to reach the correct weights has grown considerably. When we reduce the learning rate even more, our success rate has decreased and the number of steps required to reach the right weights has grown considerably.
### c.Differential ordering of the training set
Differential ordering of the training set had no effect on either our success rate or the number of steps needed to reach the correct weights. <br/>
You can see the codes of our tests in the 1A file.
### Using the perceptron for two classes that cannot be linearly seperable (1B FİLE)
Again, we created 50 points between -10 and 10 on the 6-dimensional plane. But this time we ensured that if our 3rd data in the randomlist is between -5 and 5, it is in one class and not in the other class.Thus, we have created two classes that can be separated by at least two lines or planes.
It was a data set that could not be seperable. When we train the dataset we found that it was very different from the success rate and the time it took to reach the right weights.
Since Perceptron achieved high success on linearly separable datasets, we saw that we correctly divided our dataset into two non-linearly separable groups.
Create two classes that cannot be linearly seperable, train the model, and see the results in 1B file.

## Separating twolinearly non-seperable classes with Rosenblatt's Perceptron.
When we plotted the dataset, we got the following picture. <br/>
![screenshot 47](https://user-images.githubusercontent.com/78887209/157647675-1bff73cf-18ef-4f6b-8e27-2a10651e60ab.jpg) <br/>
As you can see, we can correctly divide the data set into two classes with 3 lines.
We chose the intermediate layers of the Rosenblatt Perceptron according to these 3 lines and trained our dataset.
As can be seen from the picture, with these 3 lines, our data set should have been divided into two classes with 100% success.
The result was as we expected. You can see the codes in file 2A. <br/>
## Approximate function with Adaline
The function is as follows:
![screenshot 48](https://user-images.githubusercontent.com/78887209/157650497-d479fcc9-e1b7-4034-9639-9ec1ff5d4a01.jpg) <br/>
First of all, we created a dataset with 125 data. And we divided this dataset into 75-50 as training and test sets.When training our training set, we set our stopping criterion at 5.5 percent of the mean squared error.We determined it to be small. According to the 0.0025-0.0035 error values ​​we obtained in the test set our stopping criterion is adequate for our test set and our model looks successful.
The codes are in the file named 3.
