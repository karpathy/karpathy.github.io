---
layout: post
comments: false
title: "A New Compass and a New Map"
excerpt: "<img src='/assets/posts/medieval_map.jpg' width=100% atl='test'>Where have I been and where am I going?"
date: 2023-11-13
---

<img src="/assets/posts/medieval_map.jpg" width='100%'>

Like many mediocre students I spent much of my adult life without making any decisive career decisions. Undegrad, masters, PhD, postdoc... All default paths taken for lack of a better offer. I did all these things only to wake up one morning in my late 20s and and realise I had gotten a PhD by accident, in an area I don't care about. I accidently became a photonics engineer and I have realised, 7 years after starting my PhD in photonics, that I don't actually like photonics... Well, that's embarrassing.

What should you do when you wake up and find you have been sleep walking in the wrong direction? Stop, take a deep breath, and re-orient yorself. In my case I have been falling down and getting back up for the last 15 years. That's quite normal in the grand scheme of things, a little hustle here, a little hustle there, a little procrastination here, a little procrastination there, make slow progress in random directions thanks to Brownian motion and a weak prevailing wind. But you only have to get up and stay up once, and then all those falls no longer matter. Or if you can't stay up, learn to stay up longer between falls.

I can't remember the exact moment I decided to throw in the towel on the photonics career, but it was around the time chatGPT-4 was released. I asked it a technical question and the complexity and correctness of its answer blew me away. For the first time in my life I thought "this feels like magic". It was unreasonably good. It's insane than next token prediction can generate such complex chains of apparent reasoning. I knew immediately I needed to understand how it worked, and instead of being satisfied with a few high-level youtube infotainment videos, I wanted to get my hands dirty and build this stuff for real. I read Superintelligence in 2014 and I've been interested in AI and AI ethics since then, but now it felt like I had to do more than think and read about it. I had to get under the hood and build.

For me, if you want an impact on the future technically, basically all other technological development pales in comparison to AI. It is 1995, the internet is booming, and if you don't know how to code, you're going to miss it[^1]. I have decided that for me, this is the most important thing to work on, and everything else can wait. There is only one tiny, insignificant detail, I don't know machine learning or how to code[^2]. But knowledge and experience isn't a barrier to entry. The only barriers to entry are in your mind, all you have to do is step over them.

<div style="text-align: center;">
    <img src="/assets/posts/just_do_it.jpg" width="50%" />
</div>

## Where I'm Up To

I have a PhD in electronic engineering, but dispite that I managed to avoid any advanced maths or coding[^3] until the last year or so. During my postdoc, the extent of my python exposure and competence was installing python 2.7, and changing the value of a variable in a .py file and running it. I have a _long_ way to go.

### _Python_

I know the basic datastructures, how to use classes, how to use some of the basic libraries (matplotlib, numpy, pandas etc.). I can set up environments and do little projects in jupyter notebooks. I'm slightly beyond the "hello world" level, but not quite beyond the "tutorial - beginner project" level.

### _Neural Networks_

I understand the basic intuition behind stochasic gradient descent (SGD), multi-layer perceptron (MLP) networks and convolutional neural networks (CNNs), and how they are set up and learn through training. I've used the fast.ai library to fine-up image classification models on my own custom dataset, and deployed the model on huggingface spaces[^4].

### _Datastructures and Algorithms_

I've started learning basic datastructures like hashmaps, stacks, binary trees, etc. and basic algorithm concepts like binary search, sorting, depth-first search, sliding window etc. I have done 100 leetcode questions (partially for fun, but mainly to get more experience using python in a "toy" setting), mainly by watching videos of others explaining the problem and solution. I can currently solve some new easy questions without help, if their solutions use the datastructures and algorithms that I know (i.e. it is a simple "one or two-step" problem using binary search, hashmaps, linked lists or binary trees). I can't solve medium problems on my own yet (although I may have solved one or two after looking at the hint), but I understand the solutions, and then can implement them myself.

### _Maths_

I took a 1st year undergraduate maths module, but that was 13 years ago now (what...) and I've only had to use basic maths for most of my engineering career. I've re-learned quite a bit of basic linear algebra and calculus from Khan Academy and 3Blue1Brown.

## What I'm Doing

Here's a list of some of my main resources at the moment:

1. [Corey Schafer](https://www.youtube.com/@coreyms) - Great python tutorials, particularly the videos on classes and methods.
2. [NeetCode](https://www.youtube.com/@NeetCode) - Helping beginners navigate the intimidating (and potentially distracting) world of leetcode (+ python, datastructures and algorithms), and prepare for eventual coding interviews. His solutions are incredibly clear and easy to follow. I'm going currently 27 problems through his [150 problem roadmap](https://neetcode.io/roadmap).
3. [Programming Live with Larry](https://www.youtube.com/@Algorithmist) - If I get stuck on a daily leetcode problem and Neetcode doesn't have a solution Larry always has the solution with a good explanation.
4. [WilliamFiset](https://www.youtube.com/@WilliamFiset-videos) - I've found his graph theory videos useful, particularly the [topological sort video](https://www.youtube.com/watch?v=eL-KzMXSXXI&t=740s). I am beginning to understand recursion, but I often struggle to implement it correctly. I hope not to fall down the fascinnating but distracting graph theory rabbit hole...
5. [Errichto](https://www.youtube.com/channel/UCBr_Fu6q9iHYQCh13jmpbrg) - His [dynamic programming](https://www.youtube.com/watch?v=YBSt1jYwVfU&t=803s) and [binary search](https://www.youtube.com/watch?v=GU7DpgHINWQ&t=8s) videos have been very helpful (competitive programming and leetcode style programming could be another distracting rabbit hole, avoid!)
6. [Khan Academy](https://www.youtube.com/@khanacademy) - I have spent some time going through Khan Academy's calculus and linear algebra videos and courses (with problems) on the website. If you realise you have missed or forgotten some basics, there's no shame in going back as far as you need to go to pick up where you left off.
7. [3Blue1Brown](https://www.youtube.com/@3blue1brown/videos) - I am indebted to 3Blue1Brown for introducting me and shedding light on a dozen or so areas of maths, but particularly I have found the [Essense of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab), [Essense of calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr), [Differential equations](https://www.youtube.com/playlist?list=PLZHQObOWTQDNPOjrT6KVlfJuKtYTftqH6) and [Neural networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) playlists invaluable (special mention to the Fourier series explainer videos, [one](https://www.youtube.com/watch?v=spUNpyF58BY&list=PLZHQObOWTQDN52m7Y21ePrTbvXkPaWVSg&index=7) and [two](https://www.youtube.com/watch?v=r6sGWTCMz2k&list=PLZHQObOWTQDN52m7Y21ePrTbvXkPaWVSg)).
8. [FastAI's Practical Deep Learning for Coders 2022](https://www.youtube.com/playlist?list=PLfYUBJiXbdtSvpQjSnJJ_PmDQB_VyT5iU) - This is the number one course I have been following to learn how to become a deep learning practitioner. I'm not a coder and when I started the course I did not even know python, so each video takes a long time to process, and all the auxilliary steps (setting up hunggingface spaces, gradio, github, kaggle, paperspace, downloading datasets, inspecting data etc.) take me a lot of time. I'm enjoying the course greatly, and will try to flesh out some of the toy problems into little projects of my own.
9. [AndreJ Karpathy's Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) - This course has some real magic in it. As much as the fastai course is more practical in terms of getting an actual job, I love seeing behind the curtain and working with the raw nuts and bolts of how these systems work. Andrej takes the viewer (following along writing and toying with the code) on a journey to build and rediscover backpropogation. "When I clicked that button, something magical happened", and it does really feel like a slice of magic.

## Step 3 - The "Plan"

"Plan" is maybe too strong of a word.

---

[^1]: Obviously, if you have a particular passion for something else, then go work on that, because you won't succeed at something you're faking
[^2]: ... and LLMs are making coding obsolete, and I still have a full-time job, and I have newborn child, and I hardly any time to spend on AI or learning to code, five tiny insignificant details...
[^3]: MatLab excluded.
[^4]: [Pet classifier](https://huggingface.co/spaces/levjam/fastai_pet_classifier) (based on the fastai library, the fastai 2022 course, and the [Tanishq Abraham Gradio](https://www.tanishq.ai/blog/posts/2021-11-16-gradio-huggingface.html) tutorial); [Bear classifier](https://huggingface.co/spaces/levjam/bears_classifier) (based on a custom downloaded dataset)
