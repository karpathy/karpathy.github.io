---
layout: post
comments: true
title:  "Lessons learned from manually classifying CIFAR-10"
excerpt: "CIFAR-10 is a popular dataset small dataset for testing out Computer Vision Deep Learning learning methods. We're seeing a lot of improvements. But what is the human baseline?"
date:   2011-04-27 22:00:00
---

### CIFAR-10

> Note, this post is from 2011 and slightly outdated in some places.

<div style="text-align:center;"><img src="/assets/cifar_preview.png"></div>

**Statistics**. CIFAR-10 consists of 50,000 training images, all of them in 1 of 10 categories (displayed left). The test set consists of 10,000 novel images from the same categories, and the task is to classify each to its category. The state of the art is currently at about 80% classification accuracy (4000 centroids), achieved by [Adam Coates et al. (PDF)](http://ai.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf). This paper achieved the accuracy by using whitening, k-means to learn many centroids, and then using a soft activation function as features.

**State of the Art performance.** By the way, running their method with 1600 centroids gives 77% classification accuracy. If you set the clusters to be random the accuracy becomes 70%, and if you set the clusters to be random patches from the training set, the accuracy goes up to 74%. It seems like the whole purpose of k-means is to nicely spread out the clusters around the data. I'm guessing that the 70% random clusters performance might be because many of the clusters are relatively too far away from data manifolds, and never become activated -- it's as if you had much fewer clusters to begin with.

**Human Accuracy.** Over the weekend I wanted to see what kind of classification accuracy a human would achieve on this dataset. I set out to write some quick MATLAB code that would provide the interface to do this. It showed one image at a time and allowed me to press a key from 0-9 indicating my belief about its class category. My classification accuracy ended up at about **94%** on 400 images. Why not 100%? Because some images are really unfair! To give you an idea, here are some questionable images from CIFAR-10:
<img src="/assets/cifar_weirdimages.png">

> CIFAR-10 human accuracy is approximately 94%

### Observations
A few observations I derived from this exercise:

- The objects within classes in this dataset can be extremely varied. For example the "bird" class contains many different types of bird (both big birds and small). Not only are there many types of bird, but the occur at many possible magnifications, all possible angles and all possible poses. Sometimes only parts of the bird are shown. The poses problem is even worse for the dog/cat category, because these animals occur at many many different types of poses, and sometimes only the head is shown. Or left part of the body, etc.

- My classification method felt strangely dichotomous. Sometimes you can clearly see the animal or object and classify it based very highly-informative distinct parts (for example, you find ears of a cat). Other times, my recognition was purely based on context and the overall cues in the image such as the colors.

- The CIFAR-10 dataset is too small to properly contain examples of everything that it is asking for in the test set. I base this conclusion at least on my multiple ways of visualizing the nearest image in the training set.

- I don't quite understand how Adam Coates et al. perform so well on this dataset (80%) with their method. My guess is that it works along the following lines: looking at the image squinting your eyes you can almost always narrow down the category to about 2 or 3. The final disambiguation probably comes from finding very good specific informative patches (like a patch of some kind of fur, or pointy ear part, etc.). The k-means dictionary must be catching these cases and the SVM likely picks up on them.

- My impression from this exercise is that it will be hard to go above 80%, but I suspect improvements might be possible up to range of about 85-90%, depending on how wrong I am about the lack of training data. (**2015 update**: Obviously this prediction was way off, with state of the art now in 95%, as seen in this [Kaggle competition leaderboard](https://www.kaggle.com/c/cifar-10/leaderboard). I'm impressed!)

I encourage people to try this for themselves (see my code, above), as it is very interesting and fun! I have trouble exactly articulating what I learned, but overall I feel like I gained more intuition for image classification tasks and more appreciation for the difficulty of the problem at hand.

Finally, here is an example of my debugging interface:
<img src="/assets/cifar_predict.jpg" width="100%">

The Matlab code used to generate these results can be found [here](http://cs.stanford.edu/people/karpathy/cifar10inspect.zip)

