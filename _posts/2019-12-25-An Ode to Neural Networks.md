---
layout: post
title:  An Ode to Neural Networks
excerpt: "Unravelling Neural Networks for beginners with proper analogies and descriptions."
comments: true
---
## Motivation

I am really smart.

But just how smart really am I? I can do a lot of things, yes. Let's make a list of some of the things I can do:
- Write complete coherent sentences.
- Play badminton which requires strong hand eye coordination.
- Eat food.
- Talk.
- Shoot basketball through hoops (I'm particularly good at this).

Alright, not a bad list of innate things human bodies are capable of doing. But what is the human brain capable of doing. Let's make another list:
- Study and retain information for later use.
- Abstain from something harmful physically or mentally.
- Hold the ability to feel emotions on an unimaginably complex scale that affects almost every action we take.
- Plan about the future to the most minuscule of the details and strive to attain that particular timeline of the future in any possible way.
- Run complex calculations within milliseconds.
- And yet, fail to do [relatively] simple calculations on our math tests.

But why does that last thing happen to us. If computer scientists' claim that our brain holds more power than a supercomputer is indeed true, then why does our most supreme organ often fail to do simple tasks.

In my opinion, that is because our brain has evolved over the millenia to adapt to our way of lifestyle. Instead of being required to work like Deep Blue, our brain needed to understand more complex things, like the movement patterns of different animals, periodicity of rain and different seasons, functionality and use of different materials like wood and mud for building shelters and formulating new medicines. The necessity for crunching large numbers simply just dwindled away. But losing that valuable characteristic came along with a much more important attribute of us being able to read other people's body language and voice intonation. Apparently, evolution thought that was more important.

Look at [this](https://www.youtube.com/watch?v=ktkjUjcZid0&vl=en&ab_channel=Vsauce) video of Vsauce3's Mind Field video where Michael tried to explain **The Cognitive Tradeoff Hypothesis** discussing a similar scenario.


Several decades ago, computer scientists started experimenting with a new class of algorithms related to Artificial Intelligence. Alan Turing was the father of modern Computer Science and Machine Learning.

After the initial stage, the people started focussing on solving increasingly difficult problems. Namely, self driving cars and handwritten digit recognition. Both of these are complex tasks for a computer to carry out. They require strong cognitive systems. To solve these problems the pioneers gave rise to [neural networks](https://en.wikipedia.org/wiki/Neural_network).

In its simplicity, a neural network is a multi-perceptron algorithm taking in multiple nodes of input and learning to produce the correct output by modifying its weights. A mouthful right? Let's demystify this a little bit.

## Structure and Layers

A Neural Network looks something like this

![neural network]({{site.baseurl}}/assets/img/nn.png){: style="max-width: 99%; max-height: 99%;"}

As you can three, there are three distinct phases of a Neural Network.


Every Neural Network consists of three different type of layers- Input, Hidden, and Output layers. Let's delve into these a little bit deeper.

+ **Input Layer**: The input layer is exactly what it sounds like. This is where the user is going to feed information into the neural network. For example, if we are classifying handwritten digit images of size say 28x28 pixels, the input layer will consist of 784 neurons (more on this later), sort of like an array. Input layers can hold pixel values, sound frequencies, and any other kind of structured data or information that a model can learn from.

+ **Hidden Layer**: This is where the magic happens! The hidden layers of the Neural Network contain the weights of the program. Each neuron of each layer is connected to each neuron from the previous layer (and to the next layer) through weights. The weights are a matrix of numbers that a previous layer's values are multiplied with using matrix multiplication to obtain the values for the current layer.

+ **Output Layer**: The output layer is just like any other layer. After matrix multiplication, the output layer holds the final values of the model. In order to gain insights from this layer's values, the output layer can be run against the true *Y* value of the corresponding *X* input to calculate a cost associated with that example.

When the model is trained on tens of thousands of examples, the cumulated cost on every single example is the cost of the model for a certain iteration. After *training* the model on multiple (tens or hundreds) iterations, with the model updating its weights using an algorithm called backpropagation after each iteration, the model starts to become *smart* in the sense that its weights now know how to associate an input *X* with its output *Y* accurately.

## Perceptron

But what is a neuron? Those circles can be misleading sometimes.

A neuron in a Neural Network is based off on the biological neuron that functions in our brains. They connect with each other, and fire small electrical pulses that essentially carry messages around the brain and the rest of the body.

The Neural Network neurons function in a similar way. Each neuron is essentially a mathematical function. For example, the first neuron in the first hidden layer (I will call this neuron Bob for simplicity) basically represents the matrix multiplication operation that happens from the input layer neurons onto Bob. Therefore, Bob can be seen as just a list of numbers the size of the input layer. If the input layer has 10 neurons (or just 10 numbers), Bob takes its 10 numbers, multiplies them with the 10 numbers of the input layer and then adds the outcome to get a single value which then it stores in itself. Each neuron in the hidden layer does this.

Therefore we can simplify the matrix multiplication part and use vectorization to make the process simpler. If the input layer has I number of input neurons, and the first hidden layer has J neurons, then the matrix shape of the weights of the first hidden layer is [J, I]. So in order to find the values of the first hidden layer, all we have to do it multiply the weights of the hidden layer with the input layer neurons using matrix multiplication: [J, I] x [I, 1] = [J, 1], and we will obtain the values of the first hidden layer.

## Non-linear functions

It is important for a Neural Network to use non-linear activation functions. Otherwise, the network will just compute a linear function of the given input. The purpose of hidden layers is defeated if a linear activation function (aka no activation function) is used. This is because a composition of linear functions, is in itself a linear function. Linear activation functions are useful in linear regression problems such as predicting house prices.

**But what is an activation function?**

Activation functions are mathematical functions that the values of the hidden layer are passed through to make them into more robust...numbers. On a trivial level, the importance of activations are confusing, but as we start to view the whole network from an abstract point of view, we start to notice the usefulness of activation functions.

Most activation functions are non-linear so that the model can output probabilities of different classes in the output layers.

For example, there is the **Sigmoid Activation Function**:

![sigmoid]({{site.baseurl}}/assets/img/sigmoid.png){: style="max-width: 99%; max-height: 99%;"}

This function is useful if we want to output probabilities between just two classes.

Then there is the most famous activation function, known as **Rectified Linear Unit** or just **ReLU**.

![relu]({{site.baseurl}}/assets/img/relu.png){: style="max-width: 99%; max-height: 99%;"}

This activation function is really robust and allows the network to train a strong model.


## Conclusion

Neural Networks are not that new anymore. They are easy to make and train, but still really difficult to improve. Whether it be a computer vision task to classify handwritten digits like I have done in [this](https://github.com/ramanshsharma2806/Digit-Recognizer) GitHub repository, or a Natural Language Processing problem to generate human readable text, Neural Networks are everywhere.

If you want to learn more about Neural Networks, how to train and optimize them, I strongly suggest you take the **deeplearning.ai** specialization on Coursera taught by Dr. Andrew Ng. It is *the best* place to learn from and get started with projects! In that specialization, you will make your own Neural Networks in Python, infuse Vincent van Gogh's painting style into your pictures, generate dinosaur names from mere characters, and even see your model write Shakespeare like poems, word by word.

Drop your comments below if you'd like to share something, ask any questions or simply reach out to me at [sharmar@bxscience.edu](mailto:sharmar@bxscience.edu)!

Thank you for reading all the way to the end.
