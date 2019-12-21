---
layout: post
title:  Python for beginners tutorial (plus some tips on good practices)
comments: true
---

So let me tell you about a story. It was about four years ago, my sophomore year at [The Bronx High School of Science](1). One of my courses that semester was "Coding for All". We started with the basic turtle programs. And no, it was not like having an actual pet :(

However, I did learn a lot from that class. It is surprising how much a 10-12 week class with more than half of the kids watching Youtube and browsing Reddit can teach you. It was my first introduction to true computer science. I learnt functions, variables, scripts, ord() and chr() functions, and a couple of other things. My final project was this [Encoder Decoder game](2) which by the way is still completely functional. You can download the script, run it with a special 'key' and generate an encrypted message which you can send your friends and tell them your key for them to decrypt it. What's important is my experience with it.

## My experience with Coding for All

Spoiler alert: not as good as you think.

I mean on one hand I did just get introduced to my favorite field in all of the Sciences. Physics I love. Chemistry, meh. Biology, bedtime read.

But Computer Science, boy does it make me want to forget everything and just live in the blissful reality of bits and bytes (that was a paradoxical juxtaposition by the way)


Okay, now getting back to the point, why despite learning python for the first time, it was not a huge success for me. Turns out, I did not realize the importance of it right away. The class, not the programming language. You see, the professor just gave me the functions I needed to use. I was being fed the juicy information instead of my discovering it myself.

```python
# this part was acceptable to be told because it was not straight away intuitive
first_name = input('Enter your first name here: ')

# this part was partially unfair to be given to us.
# My professor could rather have told us to do some research,
# find out how to find this function myself

first_name = first_name[0].upper() + first_name[1:]
# makes the first letter of the name uppercase and remaining lowercase
```

You ever have that feeling when your computer science teacher just feeds you the information and you just feel awfully guilty about blindly using it. No? I do though.

And that is exactly what I felt. My junior year teacher took meticulous efforts to not let this happen. Luckily, Java has a nice documentation provided by Oracle [here][3].

But my sophomore year teacher has absolutely no qualms in spoiling his students. __Let me tell you something dear readers. There is no shortcut to learning how to code!__ Yes, after you are more than a beginner, somewhat mediocre, you can assert the claim that you justifiably use shortcuts for speed and performance, but to get there, you have to do hard grunt work. And yes, you can make the argument that I am still using the online docs. But you are never going to wake up one day and have every single Python3 function memorized, but inevitably, will have to search them.

As I was reasonably unsatisfied, I started learning by myself. I took online courses, sat in front of my laptop for hours debugging that one line of Java, but at the end also felt really great when I found the solution. And by the time junior year started, I was one of the smartest programmers in my entire year. It did not happen easily. It certainly did not happen overnight. Most of the time, this was me

![thinking]({{site.baseurl}}/assets/img/frustrated.png){: style="max-width: 99%; max-height: 99%;"}

And many times, this was me

![frustrated]({{site.baseurl}}/assets/img/thinking.png){: style="max-width: 99%; max-height: 99%;"}

So yes, it is __hard work__ and only __hard work__ that pays off in the end. Don't just go around memorizing functions, but go around searching them. The habit to go to stackoverflow is imperatively important, trust me, you will thank me later.

## College acquaintances

This note is for people who have approached me regarding programming at college.

*Q*: Can you teach me how to code?

*A*: No. There are two reasons as to why.
* First, I am not in the position to be a teacher right now. Yes, I have demonstratively displayed my prowess in programming. But I do not know how to teach. I can try (literally the original purpose of this post, keep reading, there is Python down there), but most likely I will not succeed. Besides, I am currently quite busy with other work.
* Second and more importantly, _you need to teach yourself_. No, don't scoff at me. It is possible. I am an output of about 70% self taught, 30% mentored. You have to find online resources to help yourself. Best advice, if you are an absolute beginner, you are in luck, the internet is brimming with resources these days. Take your pick, and start writing some code! If you have written a line or two before, it's time to do some online courses (this counts as self teaching because you are not in a classroom with a physical teacher, just for the sake of this post). There are numerous of them online. You never need to pay a dime to learn! Always remember that. Education should be and is free, especially in this field. Best way to go forward is to __make some projects__. Don't be scared to fail! There is a good chance you will face many setbacks. And I can help you with that, but you have to take the first step.

*Q*: How do I learn Machine Learning really fast?

*A*: You cannot. Unless you just want to learn the _train_ and _test_ functions of a model, then you might as well learn that from wikipedia. But if you want to make a career in this field, you will need:

+ __Patience__. Machine Learning is the most popular field out there right now. You must be lured to it probably out of money, top jobs at big companies like Google, Microsoft, OpenAI, and its sheer awesomeness. But if you want to make it to these things, you will have to learn a lot, which will undeniably take time.

+ __Effort and Persistence__. You will have to go to online courses (like the Stanford University one I mentioned in this [post](4)), or read [tensorflow docs](5), or work on your project for weeks, scrape data, wait for the model to train, and fail, and I kid you not, debug hundreds of lines of code. It will be scary. It will be dirty. But hear me when I say that at the end, you will come out to the top and understand it all.


Again, for my friends at SRM Ramapuram, I can help you get started, help you debug your code, but I cannot help you code. That physical action of you striking keyboard keys to type out syntax in a logical manner, will have to come from you.

## Python tutorial

I apologize I am giving off a pessimistic vibe in this post (it is entirely true though I solemnly swear on the River Styx).

But as promised, I will give a basic python tutorial now. Some notes before I start -

- If you want to follow along, I suggest downloading [PyCharm](6). It is personally my favorite and go-to editor for Python, absolutely 100% recommend it.

- The tutorial will not be too extensive, it will be a mere demonstration of the workflow of python and its control flow (if else statements) structure.

- Please leave comments at the end. It doesn't take long to do so, and helps me a ton to form my future posts. If you enjoy these blog posts, please do so.


![lets get started]({{site.baseurl}}/assets/img/yohoo.png)


I will be explaining parts of the program in comments, part outside like this. I will be borrowing code my EncoderDecoder project to specify certain things. Otherwise, I am going to be explaining things impromptu. The purpose of this tutorial is not for me to compete with Coursera and [pythonlikeyoumeanit.com](7), but for you to get a *sense* of python really fast.

##### Variables

```python
# it is very easy to make variables in Python
a = 2 # now the variable a contains the value 2

b = 48.5 # variables can also contain decimals

c = a + b + 32 # you can add variables and assign the result to
# a new variable

```

##### Functions

```python
# functions are snippets of code that can contain
# specific procedures for variables. Think of them like
# laws. They are not specific to anyone, but general to everyone.

def func():
  print('hi') # prints hi to the console

# REMEMBER: function definitions (the two lines above) have to
# come before function calls (line below)

func() # this line will look for the function 'func' in the lines ABOVE
# and if it finds it, that function will be run

def func_another(a): # here a is called a parameter
  print(a ** 2) # prints a squared
  print(a / 2) # prints quotient (can be decimal)
  print(a // 2) # prints quotient (leaves off decimal)
  print(a % 2) # prints remainder when a is divided by 2

  print(a is type(int)) # prints True if a is of type integer,
  # False otherwise


func_another(5) # runs the above function with the integer 5

```

##### Return keyword

```python
# every function in python returns something

def name():
  print('hi')

# even if you don't explicitly return something, the above function
# returns None, a default by python

def fname(a):
  return a ** 2 # returns the value of a squared

# if you ran
fname(2)
# it would not print anything because the function does not
# print anything

# instead you have to run
print(fname(2))
# to print the output

# It is common practice to use the latter method

# you can also set some parameters to be set to default in the function

def function(a, b, c=2, d=None, e=[]):
  pass # this statement keeps the function empty
       # and prevents python from raising an error

# you can call the above function like

function(a, b)
function(a=a, b=b)
function(a, b, c=234, d=183981)
function(a, b, d=[1, 2, 3], e=None, c=45.56)

# observe that parameters a and b always have to come first
# in the function definition and while calling the function
# otherwise python will eat you alive with barbecue sauce
# or possibly raise some errors

```

##### Classes and Objects

```python
# I am going to keep this part concise as it is a bit
# more advanced but I insist you try to learn this yourself
# from python's website

class Foo():

    def __init__(self, a, b):

        self.a_var = a
        self.b_var = b

        self.c_var = [1, 2, 3, 4]

    def get_a(self):
        return self.a_var

    def get_b(self):
        return self.b_var

    def get_sum(self):
        return self.a_var + self.b_var

# this is how you define a class
# notice you always need the __init__ function
# its the function that python goes to when you make a Foo() object

a, b = 2, 2 # one line shortcut
obj = Foo(a, b) # this statement goes to the __init__ function

# all class functions (including __init__ require their first parameter
# to be self)

# you instantiate class variables in the __init__ function like
# self.a = a
# self.b = []
# and you call them, in any class function, as self.var_name

print(obj.get_a()) # prints the value of a (2 in this case)

```

## Conclusion

So, I hope this short and sweet tutorial showed you just how simple
it is to code in python. What I just showed you is merely a taste of what python can do for you (Hint: anything).

If you liked this post and learned something from it, please let me know in the comments below. It will help me create my future content. **Posts request are taken**. Thank you for reading all the way to the end! :)

And if you want to see the true godly power of Python, just watch [this](8) video of David Beazley shocking everyone at PyCon India 2019 - [link](8).




[1]: https://bxscience.edu/
[2]: https://github.com/ramanshsharma2806/EncoderDecoder-game
[3]: https://docs.oracle.com/javase/8/docs/api/java/util/function/package-summary.html
[4]: https://ramanshsharma2806.github.io/blog/2019/12/18/Basics-of-Machine-Learning/
[5]: https://www.tensorflow.org/api_docs/python/tf/Tensor?version=stable
[6]: https://www.jetbrains.com/pycharm/download/#section=mac
[7]: https://www.pythonlikeyoumeanit.com/
[8]: https://www.youtube.com/watch?v=VUT386_GKI8&t=2104s&ab_channel=PythonIndia
