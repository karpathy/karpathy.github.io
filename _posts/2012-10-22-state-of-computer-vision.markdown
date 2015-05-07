---
layout: post
comments: true
title:  "The state of Computer Vision and AI: we are really, really far away."
excerpt: "A depressing look at the state of Computer Vision Research and AI in general. For those who like to think that AI is anywhere close."
date:   2012-10-22 22:00:00
---

<img src="/assets/obamafunny.jpg" width="100%" />
The picture above is funny.

But for me it is also one of those examples that make me sad about the outlook for AI and for Computer Vision. What would it take for a computer to understand this image as you or I do? I challenge you to think explicitly of all the pieces of knowledge that have to fall in place for it to make sense. Here is my short attempt:

- You recognize it is an image of a bunch of people and you understand they are in a hallway
- You recognize that there are 3 mirrors in the scene so some of those people are "fake" replicas from different viewpoints.
- You recognize Obama from the few pixels that make up his face. It helps that he is in his suit and that he is surrounded by other people with suits.
- You recognize that there's a person standing on a scale, even though the scale occupies only very few white pixels that blend with the background. But, you've used the person's pose and knowledge of how people interact with objects to figure it out.
- You recognize that Obama has his foot positioned just slightly on top of the scale. Notice the language I'm using: It is in terms of the 3D structure of the scene, not the position of the leg in the 2D coordinate system of the image.
- You know how physics works: Obama is leaning in on the scale, which applies a force on it. Scale measures force that is applied on it, that's how it works =&gt; it will over-estimate the weight of the person standing on it.
- The person measuring his weight is not aware of Obama doing this. You derive this because you know his pose, you understand that the field of view of a person is finite, and you understand that he is not very likely to sense the slight push of Obama's foot.
- You understand that people are self-conscious about their weight. You also understand that he is reading off the scale measurement, and that shortly the over-estimated weight will confuse him because it will probably be much higher than what he expects. In other words, you reason about implications of the events that are about to unfold seconds after this photo was taken, and especially about the thoughts and how they will develop inside people's heads. You also reason about what pieces of information are available to people.
- There are people in the back who find the person's imminent confusion funny. In other words you are reasoning about state of mind of people, and their view of the state of mind of another person. That's getting frighteningly meta.
-  Finally, the fact that the perpetrator here is the president makes it maybe even a little more funnier. You understand what actions are more or less likely to be undertaken by different people based on their status and identity.

I could go on, but the point here is that you've used a HUGE amount of information in that half second when you look at the picture and laugh. Information about the 3D structure of the scene, confounding visual elements like mirrors, identities of people, affordances and how people interact with objects, physics (how a particular instrument works,  leaning and what that does), people, their tendency to be insecure about weight, you've reasoned about the situation from the point of view of the person on the scale, what he is aware of, what his intents are and what information is available to him, and you've reasoned about people reasoning about people. You've also thought about the dynamics of the scene and made guesses about how the situation will unfold in the next few seconds visually, how it will unfold in the thoughts of people involved, and you reasoned about how likely or unlikely it is for people of particular identity/status to carry out some action. Somehow all these things come together to "make sense" of the scene.

It is mind-boggling that all of the above inferences unfold from a brief glance at a 2D array of R,G,B values. The core issue is that the pixel values are just a tip of a huge iceberg and deriving the entire shape and size of the icerberg from prior knowledge is the most difficult task ahead of us. How can we even begin to go about writing an algorithm that can reason about the scene like I did? Forget for a moment the inference algorithm that is capable of putting all of this together; How do we even begin to gather data that can support these inferences (for example how a scale works)? How do we go about even giving the computer a chance?

Now consider that the state of the art techniques in Computer Vision are tested on things like Imagenet (task of assigning 1-of-k labels for entire images), or Pascal VOC detection challenge (+ include bounding boxes). There is also quite a bit of work on pose estimation, action recognition, etc., but it is all specific, disconnected, and only half works. I hate to say it but the state of CV and AI is pathetic when we consider the task ahead, and when we think about how we can ever go from here to there. The road ahead is long, uncertain and unclear.  

I've seen some arguments that all we need is lots more data from images, video, maybe text and run some clever learning algorithm: maybe a better objective function, run SGD, maybe anneal the step size, use adagrad, or slap an L1 here and there and everything will just pop out. If we only had a few more tricks up our sleeves! But to me, examples like this illustrate that we are missing many crucial pieces of the puzzle and that a central problem will be as much about obtaining the right training data in the right form to support these inferences as it will be about making them. 

Thinking about the complexity and scale of the problem further, a seemingly inescapable conclusion for me is that we may also need embodiment, and that the only way to build computers that can interpret scenes like we do is to allow them to get exposed to all the years  of (structured, temporally coherent) experience we have,  ability to interact with the world, and some magical active learning/inference architecture that I can barely even imagine when I think backwards about what it should be capable of.

In any case, we are very, very far and this depresses me. What is the way forward? :( Maybe I should just do a startup. I have a really cool idea for a mobile local social iPhone app.

