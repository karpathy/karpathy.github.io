---
layout: post
comments: true
title:  "Lessons learned from manually classifying CIFAR-10"
excerpt: "CIFAR-10 is a popular dataset small dataset for testing out Computer Vision Deep Learning learning methods. We're seeing a lot of improvements. But what is the human baseline?"
date:   2011-04-27 22:00:00
---

### CIFAR-10

<div style="text-align:center;"><img src="/assets/cifar_preview.png"></div>

**Statistics**. CIFAR-10 consists of 50,000 training images, all of them in 1 of 10 categories (displayed left). The test set consists of 10,000 novel images from the same categories, and the task is to classify each to its category. The state of the art is currently at about 80% classification accuracy (4000 centroids), achieved by [Adam Coates et al. (PDF)](http://ai.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf). This paper achieved the accuracy by using whitening, k-means to learn many centroids, and then using a soft activation function as features.

**State of the Art performance.** By the way, running their method with 1600 centroids gives 77% classification accuracy. If you set the clusters to be random, the accuracy becomes 92% on train and 70% on test. And if you set the clusters to be random patches from the training set, accuracy goes up to 74% on test and about 91% on train. It seems like the entire purpose of k-means there is to nicely spread out the clusters around the data. As the number of clusters grows, randomly sampling train data converges towards that. The 70% random clusters performance must be because many of the clusters are relatively too far away from data manifolds, and never become activated -- it's as if you had much fewer clusters to begin with.

**Human Accuracy.** Over the weekend I wanted to see what kind of classification accuracy a human would achieve on this dataset. I set out to write some quick MATLAB code that would provide the interface to do this. My classification accuracy was about **94%** on 400 images. That's because some images are really unfair, but more importantly I felt I learned something about the nature of this task by explicitly going through the data myself and thinking about why I classified something as one class rather than another.

> CIFAR-10 Human Accuracy is approximately 94%

Here are some questionable images from CIFAR-10. The last one is supposed to be a boat:
<img src="/assets/cifar_weirdimages.png">

### Observations
Some observations as I was classifying the images myself:

- The objects within classes in this dataset are extremely varied. For example the "bird" class contains many different types of bird (both big birds and small). Not only are there many types of bird, but the occur at many possible magnifications, all possible angles and all possible poses. Sometimes only parts of the bird are shown. The poses problem is even worse for the dog/cat category, because these animals occur at many many different types of poses, and sometimes only the head is shown. Or left part of the body, etc.

- My classification method felt strangely dichotomous. Sometimes you can clearly see the animal or object and classify it based very highly-informative distinct parts (for example, you find ears of a cat). Other times, my recognition was purely based on context and the overall cues in the image such as the colors.

- The CIFAR-10 dataset is too small to properly contain examples of everything that it is asking for in the test set. I base this at least on my multiple ways of visualizing the nearest image in the training set.

- The classifier I found myself internally using the most was some strange model averaging. I think I search the image for very informative parts that hint strongly at a presence of one of the objects. For example, two dark dots that could indicate eyes.  Or legs of an animal like horse/deer. Separately, I would also extract global scene features. What kind of scene is this? Natural? Water? Sky? I would infer this based on clouds or waves or background type. The information then gets merged to produce object prediction. Finally, if nothing seemed to work, I predicted toad. (The toad images in this dataset are terrible. If you see a lot of brown noisy stuff, it's a toad).

- I don't quite understand how Adam Coates et al. perform so well on this dataset (80%) with the method they used. My guess is that it works along the following lines: looking at the image squinting your eyes you can almost always narrow down the category to about 2 or 3. The final disambiguation probably comes from finding very good specific informative patches (like a patch of some kind of fur, or pointy ear part, etc.)

- My impression from this exercise is that it will be hard to go above 80%, but I suspect improvements might be possible up to range of about 85-90%, depending on how wrong I am about the lack of training data.

I encourage people to try this for themselves (see my code, above), as it is very interesting and fun! I have trouble exactly articulating what I learned, but overall I feel like I gained more intuition for image classification tasks and more appreciation for the difficulty of the problem at hand.

Finally, here is an example of my debugging interface:
<img src="/assets/cifar_predict.jpg" width="100%">

The Matlab code used to generate these results can be found [here](http://cs.stanford.edu/people/karpathy/cifar10inspect.zip)

