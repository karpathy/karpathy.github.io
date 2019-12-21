---
layout: post
title:  Machine Learning for everybody
comments: true
---

It was about two years ago.

My junior year (and afterwards senior year) Computer Science teacher and my lifelong mentor used the term 'Machine Learning' while explaining something to the senior kids. I had no idea what it meant at that point of time. My first impression of it was that it had something to do with how machines learned (wow, I know right!) and hardware....stuff? But what I did not know was just how much my world was going to change because of it.

I did not do anything at that time to learn about it but when I heard that my friend from senior year went to [MIT's Beaver Works program](https://beaverworks.ll.mit.edu/CMS/bw/bwsi), I looked it up a bit. After all, who did not want a chance to attend MIT for one of the most elite summer programs for computer science and **Machine Learning**. There it came again. I conferred with my teacher and asked whether I will be able to do it, to which she replied with a smile-


> Yes, but you will lose sleep.


We laughed and I signed up for the Beaver Works online edx course. There were four modules as per I remember. First three were for python and numpy which were relatively easy (the requirements for applying to the summer program). I finished them and completed the programming assignments. But I was not able to attend the summer course because I was not a US Citizen or permanent resident. Sad story that destroyed my dreams aside, I am happy I was introduced to my true calling.


The online course gave us a few links to get started on Machine Learning ourselves. I looked around and stumbled upon [Michael Nielsen's tutorial](http://neuralnetworksanddeeplearning.com/index.html) but it might as well had been ancient greek for me.

I found some udemy courses but they were too black boxed. They started with support vector machines and basically taught-
```python
model = sklearn.svm.SVC()
model.train(X)
prediction = model.test(X_test)
```

This was not only notoriously incomprehensive but devoid of any true conceptual learning. I did not understand anything that was happening and felt largely unsatisfied.


I wanted to truly learn Machine Learning, grasp and assimilate its ideas and concepts, embody its culture, and enclose myself with it forever. Alright, excuse me for the cheesy part but it's true, I wanted to learn what it is all about but from the beginning, and at a controlled pace.

It took me some time, even tried this [hackerearth tutorial](https://www.hackerearth.com/practice/machine-learning/prerequisites-of-machine-learning/basic-probability-models-and-rules/tutorial/) but to no vain, and finally, quite accidentally, stumbled upon this [Stanford University course](https://www.coursera.org/learn/machine-learning).

The Stanford course changed my life. It begins from the absolute start, Dr. Andrew Ng starts explaining where Machine Learning is used in industries and how extremely fascinating it is.
> The 11 week course is perfect for every beginner to get started
> in Machine Learning. It requires a great deal of hard work and
> persistence, but in the end, you reap sweet rewards - Ram

It's true, this course, although painfully uses MATLAB (sorry Dr. Ng), but makes you do the grunt work and implement Machine Learning algorithms from scratch. And no, MATLAB does not have PyCharm's code completion.

Enough about my introduction to Machine Learning. Now to give you guys a taste of this, let's get started with this.

Imagine that you have some data like this,

![linear data]({{site.baseurl}}/assets/img/data.png){: style="max-width: 99%; max-height: 99%;"}

Now in order to find a good fit to the data, you need to find a best fit line, something like this,

![linear regression]({{site.baseurl}}/assets/img/lr.png){: style="max-width: 99%; max-height: 99%;"}

In this case, you would use linear regression, a most basic Machine Learning algorithm to fit linear data. Now, since actually teaching linear regression is beyond the scope of this post, I will intrigue your minds with something else.

>In Machine Learning, we try to minimize the cost based on our parameters. In even more simpler terms, you learn, you make some mistakes, and you try to reduce those mistakes as much as possible.

The cost function allows us to eyeball (if it is in 2 or 3 dimensions only) the minimum cost we would like our model (your Machine Learning program) to land on.

There are different kinds of cost functions. Some are scary like this,

![big cost function]({{site.baseurl}}/assets/img/oofgd.png){: style="max-width: 99%; max-height: 99%;"}

Or this,

![another big cost function]({{site.baseurl}}/assets/img/oofgd1.png){: style="max-width: 99%; max-height: 99%;"}


However, for linear regression, our cost function will be this,

![linear regression cost function]({{site.baseurl}}/assets/img/gd.png){: style="max-width: 99%; max-height: 99%;"}


Think of your program as the black ball on the cost function. Let's say your parameters (what you model uses to predict an answer for a new test data point) place you, the model, on some non-zero cost value on the graph. Now, you calculate your cost function's derivative and subtract and move toward's the negative direction (you always want to reduce your cost). As you do this, hopefully, you approach the minimum which consequently reduces your derivative, making you take smaller steps, ensuring that when you get to the minimum, you stay there. At this point, your model has learned as much as it can, and is ready to be used to make predictions for new data.

## Conclusion

That's it! If you understood that, no matter how trivial it seems, you understand the basic concept of Machine Learning. Just remember this *Homeric hymn*-

> The key isn’t success, it’s to fail expensively and learn from it- Robert Walker [[link](https://twitter.com/SirRobertWalker/status/1157527028628779008)]

Drop your comments below if you'd like to share something, ask any questions (_**requests for future posts are strongly considered**_) or simply reach out to me at [sharmar@bxscience.edu](mailto:sharmar@bxscience.edu)!

Thank you for reading all the way to the end.
