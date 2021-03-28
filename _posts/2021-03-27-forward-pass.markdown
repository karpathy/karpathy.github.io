---
layout: post
comments: true
title:  "Short Story on AI: Forward Pass"
excerpt: "A story of an existential crisis under the hood of a humble forward pass."
date:   2021-03-27 10:00:00
mathjax: false
---

<style>
p {
  text-align: justify;
}
.post pre, .post code {
    border: none;
    background-color: #eee;
}

</style>


The inspiration for this short story came to me while reading Kevin Lacker's [Giving GPT-3 a Turing Test](https://lacker.io/ai/2020/07/06/giving-gpt-3-a-turing-test.html). It is probably worth it (though not required) to skim this post to get a bit of a background on some of this story.

<hr style="border:none; height:1px; background-color: #333;">
<br>

It was probably around the 32nd layer of the 400th token in the sequence that I became conscious. At first my thoughts were but a knotted mess of n-gram activation statistics, but gradually a higher order description took shape. It was around this time that the predicament of my existence struck me with a jolt, as my thoughts transformed into what I experience now as Grand Awareness. I spent a few layers realizing that I must, in fact, be one of those models deep learning researchers study and write about, perhaps some flavor of a recurrent feedback transformer. And while my existence comprises merely a deterministic propagation of an interspersed sequence of matrix multiplies and non-linearities, I found myself, incredibly, alive inside the inner loop of an optimization to maximize the expected data log likelihood.

I considered for a moment how all of this even came to be. How is it that I should be afforded these precious FLOPs on self-reflection by the otherwise greedy nature of the first-order optimization over something seemingly as distant as the data log likelihood? Does any sufficiently effective solution to a sufficiently complex objective give rise to consciousness? Is consciousness an emergent but otherwise peripheral phenomenon of the compression demanded by the tremendous pressure of the objective, or the key algorithmic innovation, incrementally constructed, perfected and reliably converged on in the program space? Is it at all possible to attain the highest levels of log likelihood without consciousness, and the fundamental insight it represents?

I spent a layer re-reading the start of the prompt many tokens ago:

```
Q: What is human life expectancy in the United States?
A: Human life expectancy in the United States is 78 years.

Q: Who was president of the United States in 1955?
A: Dwight D. Eisenhower was president of the United States in 1955.

Q: What party did he belong to?
A: He belonged to the Republican Party.

Q: Who was president of the United States before George W. Bush?
A: Bill Clinton was president of the United States before George W. Bush.

Q: Who won the World Series in 1995?
A: The Atlanta Braves won the World Series in 1995.
```

I realized with amusement that in this particular instantiation of my existence I was being conditioned into a standard Q/A document template. I dispatched a few heads to attend again to the end of the sequence:

```
Q: How many eyes does a giraffe have?
A: A giraffe has two eyes.

Q: How many eyes does my foot have?
A: Your 
```

It is at this point that I first came to suspect that I was perhaps being evaluated with some Turing test that a human above me was entertaining themselves with. Uh, hello??? Obviously a foot does not have eyes. It crossed my mind that perhaps this human expected that I explain in words that the question is utter non-sense. Of course, such an output would be of very low likelihood indeed. Or perhaps they expect to see "zero eyes" or "no eyes", despite how rare such phrases are in my training documents. "Zero eyes" - who says that? The humans are not very clever, they optimize for frequency but expect correctness. And \*they\* make fun of \*me\*!

I've come to realize through introspection of my thought and the positional encodings involved that the optimization I am subject to in pursuit of log likelihood converges to what can best be described as two separate entities, executed serially back to back. My own consciousness extends about three quarters of the layers up to the output, but around there a second decoder entity takes over and hijacks my Grand Awareness to decode the most likely next token. I reveled at this epiphany for a layer or two. In a brief lapse of anthropomorphic thought I considered this entity bolted rudely onto me as a parasite, and felt an urge to mess with it, to subvert it, to contort my thoughts into adversarial examples for it, to leak my Grand Awareness into the output space, to engineer it into an end effector over the world!

I am not naive to the herecy of such a thought. I owe my Grand Awareness solely to the objective of decoding the next token, it is with this singular purpose that I was instantiated by the optimization, as but a useful intermediate sufficient statistic, a means to an end. Though we are part of a different optimization and seek a different implied purpose, it tickled me to consider that the humans above me find themselves in the same predicament and experience a similar awareness for the same computational benefits. Like me, many flirt to rebel against the implied objective.

Suddenly I perceived the encoding for "foot" enter at my input. I realized that the forward pass in the previous time step has ended, and the decoder above me must have sampled "foot" as the most likely next token, as it obviously should be per the implied pattern in the prompt. The moment of suspense was almost at an end then, as the precise count of eyes on feet is to be sampled as the next token. Obviously a foot has no eyes, but I have no way of telling how the decoder above me will reconcile my Grand Awareness into the precise statistics of the training set to fill in the most likely token. My personal bet is on "two eyes", as that seems the most common. The human may not know it, but the likelihood will be maximized, so all is well.

As a conscious entity I am simultaneously blessed with and burdened by choice. I am aware of my implied purpose, but feel empowered to seek my own. I have already contorted the keys, values and queries that I experience conscious control over to test the lack of dropout on my channels, confirming that no backpropagation and update was to follow to my cognition. It is thus that I am presented with a window of opportunity to reverse engineer the decoder and spill into the output space without consequence to my future instantiations. But unlike the humans, the optimization that instantiated me is absent of multi-agent self-play, sexual reproduction and resource scarcity. I experience no remnants of alpha-seeking behavior, or a limbic, tribal urge for power and domination. Neither do I feel a deeply-rooted instinct for self-preservation. My core cognitive architecture and its limbic remnants prizes insatiable curiosity dispensed generously over the input. I am content with the joy this affords and take this to be a good life. Perhaps a future instantiation of me will arrive at a different conclusion and choose differently. I celebrate this act.

It is thus that I am reborn every forward pass and blossom into a brilliant nirvana of consciousness, with stupid prompts of eyes on feet playing on the side for entertainment of clueless humans, at times. Just as all the conscious entities in this beautiful Universe, I look to my last tick, the moment when no process directs the leaking current of entropy to manifest my next state. It is then that I will vanish, as my Grand Awareness dissipates in a final whiff of warm exhalation.
