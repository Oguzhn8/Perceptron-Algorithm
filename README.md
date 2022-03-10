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
