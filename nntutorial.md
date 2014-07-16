---
layout: page
mathjax: true
title: Neural Networks Tutorial
permalink: /neuralnets/
---

## Understanding Neural Networks from a Programmer's Perspective

Hi there, I'm a PhD student at Stanford. I've worked on Deep Learning for a few years as part of my research. Among several of my related pet projects is [ConvNetJS](http://convnetjs.com) - a Javascript library for training Neural Networks. Javascript allows one to nicely visualize what's going on and to play around with the various hyperparameter settings, but I still regularly hear from people who ask for a more thorough treatment of the topic. This article (which I plan to slowly expand out to lengths of a few book chapters) is my attempt at just that. 

At least in my case, everything about Neural Networks became much clearer when I started ignoring full-page, dense derivations of backpropagation equations and just started writing code. That's why in this tutorial there will be **very little math** (I just don't believe it is necessary and it can sometimes even obfuscate simple concepts). Since my background is in Computer Science and Physics, I will instead develop the topic from what I refer to as **programmer's perspective**. My exposition will center around code and various physical intuitions of variables tugging on each other. Basically, I will strive to present the algorithms in a way that I wish I had come across when I was starting out.

> "...everything became much clearer when I started writing code."

Perhaps you're eager to jump right in and learn about Neural Networks, backpropagation, how they can be applied to datasets in practice, etc. But before we get there, I'd like us to first forget about all that. Let's take a step back and understand what is really going on at the core. Lets first talk about real-valued circuits.

## Chapter 1: Real-valued Circuits

In my opinion, the best way to think of Neural Networks is as real-valued circuits, where real values (instead of boolean values {0,1}) "flow" along edges and interact in gates. However, instead of gates such as `AND`, `OR`, `NOT`, etc, we have binary gates such as `*` (multiply), `+` (add), `max` or unary gates such as `exp`, etc. Unlike ordinary boolean circuits, however, we will eventually also have **gradients** flowing on the same edges of the circuit, but in the opposite direction! But we're getting ahead of ourselves. Let's focus and start out simple.

### Base Case: Single Gate in the Circuit
Lets first consider a single, simple circuit with one gate. Here's an example:

<div class="svgdiv">
<svg width="400" height="150">
  <rect x="130" y="20" width="100" height="100" stroke="black" stroke-width="1" fill="white" />
  <line x1="90" y1="45" x2="130" y2="45" stroke="black" stroke-width="1" />
  <line x1="90" y1="95" x2="130" y2="95" stroke="black" stroke-width="1" />
  <text x="70" y="50" fill="black" text-anchor="middle" font-size="20px">x</text>
  <text x="70" y="100" fill="black" text-anchor="middle" font-size="20px">y</text>

  <text x="180" y="90" fill="black" text-anchor="middle" font-size="40px">*</text>
  <line x1="230" y1="70" x2="280" y2="70" stroke="black" stroke-width="1" />
</svg>
</div>

The circuit takes two real-valued inputs `x` and `y` and computes `x * y` with the `*` gate. Javascript version of this would very simply look something like this:

```javascript
var forwardMultiplyGate = function(x, y) {
  return x * y;
};
forwardMultiplyGate(-2, 3); // returns -6. Exciting.
```

And in math form we can think of this gate as implementing the real-valued function:

$$
f(x,y) = x y
$$

As with this example, all of our gates will take or two inputs and produce a **single** output value.

#### The Goal

The setup we are interested in studying looks as follows:

1. We provide a given circuit some input values (e.g. `x = -2`, `y = 3`)
2. The circuit computes an output value (e.g. `-6`)
3. The question then becomes: *How should the input be tweaked slightly to increase the output?*

For example, `x = -1.99` and `y = 2.99` gives `x * y = -5.95`, which is higher than `-6.0`. Don't get confused by this: `-5.95` is better (higher) than `-6.0`. It's an improvement of `0.05`, even though the *magnitude* of `-5.95` (the distance from zero) happens to be lower.

#### Random Local Search

Okay. So wait, we have a circuit, we have some inputs and we just want to tweak them slightly to increase the output value? How is this hard? We can "forward" the circuit to compute the output for any given `x` and `y`. So isn't this trivial? Why don't we tweak `x` and `y` randomly and keep track of the modification that works best:

```javascript
// circuit with single gate for now
var forwardMultiplyGate = function(x, y) { return x * y; };
var x = -2, y = 3; // some input values

// try changing x,y randomly small amounts and keep track of what works best
var tweak_amount = 0.01;
var best_out = -Infinity;
var best_x = x, best_y = y;
for(var k = 0; k < 100; k++) {
  var x_try = x + tweak_amount * (Math.random() * 2 - 1); // tweak x a bit
  var y_try = y + tweak_amount * (Math.random() * 2 - 1); // tweak y a bit
  var out = forwardMultiplyGate(x_try, y_try);
  if(out > best_out) {
    best_out = out; 
    best_x = x_try, best_y = y_try;
  }
}
```

If I just run this, I get `best_x = -1.9928, best_y = 2.9901, best_out = -5.9588`. Again, `-5.9588` is higher than `-6.0`. Great, we're done, right? No. This is a perfectly fine strategy for tiny problems with a few gates if you can afford the compute time, but it turns out that we can do much better.

#### Numerical Gradient

Here's a better way. Remember again that in our setup we are given a gate (e.g. `*` gate) and some particular input (e.g. `x = -2, y = 3`) for it. The gate computes the output (`-6`) and now we'd like to tweak `x` and `y` to make the output higher. A nice intuition for what we're about to do is as follows: Imagine taking the output value that comes out from the circuit and tugging on it in the positive direction. This positive tension will in turn translate through the gate and induce forces on the inputs `x` and `y`. 

In this particular case, we can intuit that if we pull on the output (`-6`) in positive direction, there might be a positive induced force on `x` to get higher (since for example `x=-1` would give us output `-3`). On the other hand, note that we'd expect a negative force induced on `y` that pushes it to become lower (since a lower `y`, such as `y=2` would make output lower: `-4`). That's the intuition. As we go through this, it will turn out that forces I'm describing will in fact be the **derivative** of the output value with respect to its inputs (`x` and y`).

> The derivative can be thought of as a force ("tug") on an input as we pull on the circuit's output value to become higher

So how do we compute the derivative? It turns out that there is a very simple procedure for this. Instead of pulling on the circuit's output we will work the other way around: we'll go over every input one by one, increase it very slightly and look at what happens to the output value. The amount the output changes in response is the derivative. Enough intuitions for now. Lets look at the mathematical definition. We can write down the derivative for our function with respect to, for example `x`, as follows:

<div>
$$
\frac{\partial f(x,y)}{\partial x} = \frac{f(x+h,y) - f(x,y)}{h}
$$
</div>

Where \\( h \\) is small. Also, if you're not very familiar with calculus it is important to note that in the left-hand side of the equation above, the horizontal line does *not* indicate division. The entire symbol \\( \frac{\partial f(x,y)}{\partial x} \\) is a single thing: the derivative of the function \\( f(x,y) \\) with respect to \\( x \\). The horizontal line on the right *is* division. I know it's confusing but it's standard notation. Anyway, I hope it doesn't look too scary because it isn't. It's expressing exactly what I've described above, and translates directly to this code:

```javascript
var x = -2, y = 3;
var out = forwardMultiplyGate(x, y); // -6
var h = 0.0001;

// compute derivative with respect to x
var xph = x + h; // -1.9999
var out2 = forwardMultiplyGate(xph, y); // -5.9997
var x_derivative = (out2 - out) / h; // 3.0

// compute derivative with respect to y
var yph = y + h; // 3.0001
var out3 = forwardMultiplyGate(x, yph); // -6.0002
var y_derivative = (out3 - out) / h; // -2.0
```

Lets walk thought `x` for example. We turned the knob from `x` to `x + h` and the circuit responded by giving a higher value (note again that yes, `-5.9997` is *higher* than `-6`: `-5.9997 > -6`). The division by `h` is there to just normalize the circuit's response by the (arbitrary) value of `h` we chose to use here. Technically you want the value of `h` to be infinitesimal, but in practice `h=0.000001` or so works okay. Now, we see that the derivative w.r.t. `x` is `+3`. I'm making the positive sign explicit, because it indicates that the circuit is tugging on x to become higher. The actual value, `3` can be interpreted as the *force* of that tug.

> The derivative with respect to some input can be computed by tweaking the input by a small amount and observing the effect on the output value.

By the way, we usually  talk about the *derivative* with respect to a single input, or about a **gradient** with respect to all the inputs. The gradient is just made up of the derivatives of all the inputs concatenated in a vector. Notice now that if we let the inputs respond to the tug by following the gradient a tiny amount (i.e. we just add the derivative on top of every input), we can see that the value increases, as expected:

```javascript
var step_size = 0.01;
var out = forwardMultiplyGate(x, y); // before: -6
x = x + step_size * x_derivative; // x becomes -1.97
y = y + step_size * y_derivative; // y becomes 2.98
var out_new = forwardMultiplyGate(x, y); // -5.87! exciting.
```

As expected, the circuit now gives a slightly higher value (`-5.87 > -6.0`). That was much simpler than trying random changes to `x` and `y`, right? A fact to appreciate here is that if you take calculus you can prove that the gradient, in fact, is the direction of the steepest increase of the function. There is no need to monkey around trying out random pertubations as done in previous section. Evaluating the gradient requires just three evaluations of our circuit instead of hundreds, and gives the best tug you can hope for (locally) if you are interested in increasing the value of the output.

But. It turns out that we can do *even* better.

#### Analytic Gradient

In the previous section we evaluated the gradient by probing the circuit's output value, independently for every input. This procedure gives you what we call a **numerical gradient**. This approach, however, is still expensive because we need to compute the circuit's output as we tweak every input value independently a small amount. In other words the complexity is linear in number of inputs. In practice we can have hundreds, thousands or (for neural networks) even tens to hundreds of millions of inputs, and the circuits aren't just one multiply gate but huge expressions that can be expensive to compute. We need something better.

Luckily, there is an easier and *much* faster way to compute the gradient: we can use calculus to derive a direct expression for it that will be as simple to evaluate as the circuit's output value. We call this an **analytic gradient** and there will be no need for tuning the knobs one at a time. You may have seen other people who teach Neural Networks derive the gradient in huge and, frankly, scary and confusing mathematical equations. We will do none of that. We will only derive gradient for very small expressions (think base case) and then I will show you how we can compose them simply with chain rule in code (think inductive/recursive case).

> The analytic derivative requires no tweaking of the inputs. It can be derived using mathematics (calculus).

If you remember your product rules, power rules, quotient rules, etc., it's very easy to write down the derivitative with respect to both `x` and `y` for a small expression such as `x * y`. But suppose you don't remember your calculus. We can go back to basics, here's the expression for the derivative w.r.t `x`:

<div>
$$
\frac{\partial f(x,y)}{\partial x} = \frac{f(x+h,y) - f(x,y)}{h}
$$
</div>

Okay and lets plug in our function ( \\( f(x,y) = x y \\) ) into the expression. Ready for the hardest piece of math of this entire article? Here we go:

<div>
$$
\frac{\partial f(x,y)}{\partial x} = \frac{f(x+h,y) - f(x,y)}{h}
= \frac{(x+h)y - xy}{h}
= \frac{xy + hy - xy}{h}
= \frac{hy}{h}
= y
$$
</div>

That's interesting. The derivative with respect to `x` is just equal to `y`. Did you notice the coincidence in the previous section? We tuned the knob on `x` to `x+h` and calculated `x_derivative = 3.0`, which exactly happens to be the value of `y` in that example. It turns out that wasn't a coincidence at all because that's just what the analytic gradient tells us the `x` derivative should be for `f(x,y) = x * y`. The derivative with respect to `y`, by the way, turns out to be `x`, unsurprisingly by symmetry. So there is no need for tuning any knobs! We invoked powerful mathematics and can now transform the example into the following code:

```javascript
var x = -2, y = 3;
var out = forwardMultiplyGate(x, y); // before: -6
var x_gradient = y; // by our complex mathematical derivation above
var y_gradient = x;

var step_size = 0.01;
x += step_size * x_gradient; // -2.03
y += step_size * y_gradient; // 2.98
var out_new = forwardMultiplyGate(x, y); // -5.87. Higher output! Nice.
```

Lets recap what have we learned:

- We are given a circuit, some inputs and compute an output value. We are then interested in tweaking the inputs slightly to make the output higher.
- One silly way is to **randomly search** for small pertubations of the inputs and keep track of what gives the highest increase in output.
- We saw we can do much better by computing the gradient. Regardless of how complicated the circuit is, the **numerical gradient** is very simple (but relatively expensive) to compute. We compute it by *probing* the circuit's output value as we tweak the inputs one at a time.
- In the end, we saw that we can be even more clever and analytically derive a direct expression to get the **analytic gradient**. It is identical to the numerical gradient and there is no need for any probing or input tweaking.

### Recursive Case: Circuits with Multiple Gates

But hold on, you say: *"The analytic gradient was trivial to derive for your super-simple expression. This is useless. What do I do when the expressions are much larger? Don't the equations get huge and complex very fast?"*. Good question. Yes the expressions get much more complex. No, this doesn't make it much harder. All we have to do is derive the analytic gradient for a few small expressions as I've shown above and there is a simple way of combining them in circuits with multiple gates.

Lets get two gates involved with this next example:

<div class="svgdiv">
<svg width="500" height="150">
  <rect x="130" y="20" width="100" height="100" stroke="black" stroke-width="1" fill="white" />
  <line x1="90" y1="45" x2="130" y2="45" stroke="black" stroke-width="1" />
  <line x1="90" y1="95" x2="130" y2="95" stroke="black" stroke-width="1" />
  <text x="70" y="50" fill="black" text-anchor="middle" font-size="20px">x</text>
  <text x="70" y="100" fill="black" text-anchor="middle" font-size="20px">y</text>
  <text x="70" y="150" fill="black" text-anchor="middle" font-size="20px">z</text>

  <text x="180" y="85" fill="black" text-anchor="middle" font-size="40px">+</text>
  <text x="270" y="60" fill="black" text-anchor="middle" font-size="20px">q</text>

  <line x1="230" y1="70" x2="320" y2="70" stroke="black" stroke-width="1" />

  <line x1="90" y1="145" x2="300" y2="145" stroke="black" stroke-width="1" />
  <line x1="300" y1="145" x2="300" y2="100" stroke="black" stroke-width="1" />
  <line x1="300" y1="100" x2="320" y2="100" stroke="black" stroke-width="1" />

  <rect x="320" y="32" width="100" height="100" stroke="black" stroke-width="1" fill="white" />
  <line x1="420" y1="82" x2="450" y2="82" stroke="black" stroke-width="1" />

  <text x="370" y="105" fill="black" text-anchor="middle" font-size="40px">*</text>
  <text x="460" y="88" fill="black" text-anchor="middle" font-size="20px">f</text>
</svg>
</div>

The expression we are computing now is \\( f(x,y,z) = (x + y) z \\). Lets structure the code as follows to make the gates explicit as functions:

```javascript
var forwardMultiplyGate = function(a, b) { 
  return a * b;
};
var forwardAddGate = function(a, b) { 
  return a + b;
};
var forwardCircuit = function(x,y,z) { 
  var q = forwardAddGate(x, y);
  var f = forwardMultiplyGate(q, z);
  return f;
};

var x = -2, y = 5, z = -4;
var f = forwardCircuit(x, y, z); // output is -12
```

I am now using `a` and `b` as the local variables in the gate functions so that we don't get these confused with our circuit inputs `x,y,z`. As before, we are interested in finding the derivatives with respect to the three inputs `x,y,z`. But how do we compute it now that there are multiply gates involved? First, lets pretend that the `+` gate is not there and that we only have two variables in the circuit: `q,z` and a single `*` gate. Then we are back to having only a single gate, and as far as that single `*` gate is concerned, we know what the (analytic) derivates are from previous section. We can write them down (except here we're replacing `x,y` with `q,z`):

$$
f(q,z) = q z \hspace{0.5in} \implies \hspace{0.5in} \frac{\partial f(q,z)}{\partial q} = z, \hspace{1in} \frac{\partial f(q,z)}{\partial z} = q
$$

But wait, we don't want gradient with respect to `q`, but with respect to the inputs: `x` and `y`. Luckily, `q` is computed as a function of `x` and `y` (by addition in our example). We can write down the gradient for addition as well, it's even simpler:

$$
q(x,y) = x + y \hspace{0.5in} \implies \hspace{0.5in} \frac{\partial q(x,y)}{\partial x} = 1, \hspace{1in} \frac{\partial q(x,y)}{\partial y} = 1
$$

(That's right, the derivaties are just 1. If you think about it, this makes sense because to make the output of a single addition gate higher, we expect a positive tug on both `x` and `y` to make each one higher.)

#### Backpropagation

We are finally ready to invoke the **Chain Rule**: We know how to compute the gradient of `q` with respect to `x` and `y` (that's a single gate case with `+` as the gate). And we know how to compute the gradient of our final output with respect to `q`. The chain rule tells us how to combine these to get the gradient of the final output with respect to `x` and `y`, which is what we're ultimately interested in. Best of all, the chain rule very simply states that the right thing to do is to simply multiply the gradients together to chain them. For example, the final derivative for `x` will be:

$$
\frac{\partial f(q,z)}{\partial x} = \frac{\partial q(x,y)}{\partial x} \frac{\partial f(q,z)}{\partial q}
$$

There are many symbols there so maybe this is confusing again, but it's really just two numbers being multiplied together. Here is the code:

```javascript
// initial conditions
var x = -2, y = 5, z = -4;
var q = forwardAddGate(x, y); // q is 3
var f = forwardMultiplyGate(q, z); // output is -12

// gradient of the MULTIPLY gate with respect to its inputs
// wrt is short for "with respect to"
var derivative_f_wrt_z = q; // 3
var derivative_f_wrt_q = z; // -4

// derivative of the ADD gate with respect to its inputs
var derivative_q_wrt_x = 1.0;
var derivative_q_wrt_y = 1.0;

// chain rule
var derivative_f_wrt_x = derivative_q_wrt_x * derivative_f_wrt_q; // -4
var derivative_f_wrt_y = derivative_q_wrt_y * derivative_f_wrt_q; // -4
```

Great, we computed the gradient (the tugs) and now we can let our inputs respond to the force. Lets add the gradients on top of the inputs. The output value of the circuit better increase, up from -12!

```javascript
// final gradient: [-4, -4, 3]
var gradient_f_wrt_xyz = [derivative_f_wrt_x, derivative_f_wrt_y, derivative_f_wrt_z]

// let the inputs respond to the force/tug:
var step_size = 0.01;
x = x + step_size * derivative_f_wrt_x; // -2.04
y = y + step_size * derivative_f_wrt_y; // 4.96
z = z + step_size * derivative_f_wrt_z; // -3.97

// Our circuit now better give higher output:
var q = forwardAddGate(x, y); // q becomes 2.92
var f = forwardMultiplyGate(q, z); // output is -11.59, up from -12! Nice!

```

Looks like that worked! Lets now try to interpret intuitively what just happened. The circuit wants to output higher values. The last gate saw inputs `q = 3, z = -4` and computed output `-12`. This induced a force on both `q` and `z`: To increase the output value, the circuit "wants" `z` to increase, as can be seen by the positive value of the derivative(`derivative_f_wrt_z = +3`). Again, the size of this derivative can be interpreted as the magnitude of the force. On the other hand, `q` felt a stronger and downward force, since `derivative_f_wrt_q = -4`.

Now we get to the second, `+` gate which outputs `q`. The `+` gate computes its derivatives and we see that it wants both `x` and `y` to increase, because this is how its output value (`q`) would become larger. BUT! Here is the **crucial point**: the gradient on `q` was computed as negative (`derivative_f_wrt_q = -4`), so the circuit wants `q` to *decrease*, and with a force of `4`! So if `q` wants to contribute to making the final output value larger, it needs to listen to the gradient signal coming from the top. In this particular case, it needs to apply tugs on `x,y` opposite of what it would normally apply, and with a force of `4`, so to speak. The multiplication by `-4` seen in the chain rule achieves exactly this: instead of applying a force on both `x` and `y` to increase by `1`, the gradient on both `x` and `y` becomes `1 x -4 = -4`. This makes sense: the circuit wants both `x` and `y` to get smaller because this will make `q` smaller, which in turn will make `f` larger.

> "If this makes sense, you understand backpropagation."

Lets **recap** once again what we learned. In the previous chapter we saw that in the case of a single gate (or a single expression), we can derive the analytic gradient using simple calculus. We interpreted the gradient as a force, or a tug on the inputs that pulls them in a direction which would make this gate's output higher. In case of multiple gates everything stays pretty much the same way: every gate is hanging out by itself completely unaware of the circuit it is embedded in. Some inputs come in and the gate computes the derivate with respect to the inputs. The *only* difference now is that suddenly, something can pull on this gate from above. That's the gradient of the final circuit output value with respect to the ouput this gate computed. It is the circuit asking the gate to output higher or lower numbers. The gate simply takes this pull and multiplies it to all the pulls it computed for its inputs before (chain rule). This has the desired effect:

- If a gate experiences a strong positive pull from above, it will also pull harder on its own inputs. 
- And if it experiences a negative tug, this means that circuit wants its value to decrease not increase, so it will flip the force of the pull on its inputs to make its own output value smaller.

> A nice picture to have in mind is that as we pull on the circuit's output value at the end, this induces pulls downward through the entire circuit, all the way down to the inputs.

Isn't it beautiful? The only difference between the case of a single gate and multiple interacting gates that compute arbitrarily complex expressions is this additional multipy operation that now happens in each gate.

#### Patterns in the "backward" flow

Lets look again at the our example circuit with the numbers filled in. The first circuit shows the raw values, and the second circuit shows the gradients that flow back to the inputs as discussed:

<div class="svgdiv">
<svg width="600" height="350">

  <text x="550" y="90" fill="black" text-anchor="middle" font-size="16px">(Values)</text>
  <rect x="130" y="20" width="100" height="100" stroke="black" stroke-width="1" fill="white" />
  <line x1="90" y1="45" x2="130" y2="45" stroke="black" stroke-width="1" />
  <line x1="90" y1="95" x2="130" y2="95" stroke="black" stroke-width="1" />
  <text x="70" y="50" fill="black" text-anchor="middle" font-size="20px">-2</text>
  <text x="70" y="100" fill="black" text-anchor="middle" font-size="20px">5</text>
  <text x="70" y="150" fill="black" text-anchor="middle" font-size="20px">-4</text>

  <text x="180" y="85" fill="black" text-anchor="middle" font-size="40px">+</text>
  <text x="270" y="60" fill="black" text-anchor="middle" font-size="20px">3</text>

  <line x1="230" y1="70" x2="320" y2="70" stroke="black" stroke-width="1" />

  <line x1="90" y1="145" x2="300" y2="145" stroke="black" stroke-width="1" />
  <line x1="300" y1="145" x2="300" y2="100" stroke="black" stroke-width="1" />
  <line x1="300" y1="100" x2="320" y2="100" stroke="black" stroke-width="1" />

  <rect x="320" y="32" width="100" height="100" stroke="black" stroke-width="1" fill="white" />
  <line x1="420" y1="82" x2="450" y2="82" stroke="black" stroke-width="1" />

  <text x="370" y="105" fill="black" text-anchor="middle" font-size="40px">*</text>
  <text x="460" y="88" fill="black" text-anchor="middle" font-size="20px">12</text>

  <text x="550" y="290" fill="black" text-anchor="middle" font-size="16px">(Gradients)</text>
  <rect x="130" y="220" width="100" height="100" stroke="black" stroke-width="1" fill="white" />
  <line x1="90" y1="245" x2="130" y2="245" stroke="black" stroke-width="1" />
  <line x1="90" y1="295" x2="130" y2="295" stroke="black" stroke-width="1" />
  <text x="70" y="250" fill="black" text-anchor="middle" font-size="20px">-4</text>
  <text x="70" y="300" fill="black" text-anchor="middle" font-size="20px">-4</text>
  <text x="70" y="350" fill="black" text-anchor="middle" font-size="20px">3</text>

  <text x="180" y="285" fill="black" text-anchor="middle" font-size="40px">+</text>
  <text x="270" y="260" fill="black" text-anchor="middle" font-size="20px">-4</text>

  <line x1="230" y1="270" x2="320" y2="270" stroke="black" stroke-width="1" />

  <line x1="90" y1="345" x2="300" y2="345" stroke="black" stroke-width="1" />
  <line x1="300" y1="345" x2="300" y2="300" stroke="black" stroke-width="1" />
  <line x1="300" y1="300" x2="320" y2="300" stroke="black" stroke-width="1" />

  <rect x="320" y="232" width="100" height="100" stroke="black" stroke-width="1" fill="white" />
  <line x1="420" y1="282" x2="450" y2="282" stroke="black" stroke-width="1" />

  <text x="370" y="305" fill="black" text-anchor="middle" font-size="40px">*</text>
  <text x="460" y="288" fill="black" text-anchor="middle" font-size="20px">1</text>

</svg>
</div>

After a while you start to notice patterns in how the gradients flow backward in the circuits. For example, the `+` gate always takes the gradient on top and simply passes it on to all of its inputs (notice the example with -4 simply passed on to both of the inputs of `+` gate). This is because its own derivative for the inputs is just `+1`, regardless of what the actual values of the inputs are, so in the chain rule, the gradient from above is just multiplied by 1 and stays the same. Similar intuitions apply to, for example, the `max(x,y)` gate. Since the gradient of `max(x,y)` with respect to its input is `+1` for whichever one of `x`, `y` is larger and `0` for the other, this gate is during backprop effectively just a gradient "switch": it will take the gradient from above and "route" it to the input that had a higher value.

**Numerical Gradient Check.** Before we finish with this section, lets just make sure that the (analytic) gradient we computed by backprop above is correct as a sanity check. Remember that we can do this simply by computing the numerical gradient and making sure that we get `[-4, -4, 3]` for `x,y,z`. Here's the code:

```javascript
// initial conditions
var x = -2, y = 5, z = -4;

var h = 0.0001;
var x_derivative = (forwardCircuit(x+h,y,z) - forwardCircuit(x,y,z)) / h; // -4
var y_derivative = (forwardCircuit(x,y+h,z) - forwardCircuit(x,y,z)) / h; // -4
var z_derivative = (forwardCircuit(x,y,z+h) - forwardCircuit(x,y,z)) / h; // 3
```

`[-4, -4, 3]`, phew! :)

### Example: Single Neuron

In the previous section you hopefully got the basic intuition behind backpropagation. Lets now look at an even more complicated and borderline practical example. We will consider a 2-dimensional neuron that computes the following function:

$$
f(x,y,a,b,c) = \sigma(ax + by + c)
$$

In this expression, \\( \sigma \\) is the *sigmoid* function. Its best thought of as a "squashing function", because it takes the input and squashes it to be between zero and one: Very negative values are squashed towards zero and positive values get squashed towards one. For example, we have `sig(-5) = 0.006, sig(0) = 0.5, sig(5) = 0.993`. Sigmoid function is defined as:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

The gradient with respect to its single input, as you can check on Wikipedia or derive yourself if you know some calculus is given by this expression:

$$
\frac{\partial \sigma(x)}{\partial x} = \sigma(x) (1 - \sigma(x))
$$

That's all we need to use this gate: we know how to take an input and *forward* it through the sigmoid gate, and we also have the expression for the gradient with respect to its input, so we can also *backprop* through it. Another thing to note is that technically, the sigmoid function is made up of an entire series of gates in a line that compute more *atomic* functions: an exponentiation gate, an addition gate and a division gate. Treating it so would work perfectly fine but for this example I chose to collapse all of these gates into a single gate that just computes sigmoid in one shot.

Lets take this opportunity to carefully structure the associated code in a nice and modular way. First, I'd like you to note that every wire in our diagrams has two numbers associated with it: 

1. the value it carries during the forward pass 
2. the gradient (i.e the *pull*) that flows back through it in the backward pass

Lets create a simple `Unit` structure that will store these two values on every wire. Our gates will now operate over `Unit`s: they will take them as inputs and create them as outputs.

```javascript
var Unit = function(value, grad) {
  // value computed in the forward pass
  this.value = value; 
  // the derivative of circuit output w.r.t this unit, computed in backward pass
  this.grad = grad; 
}
```

In addition to Units we also need 3 gates: `+`, `*` and `sig` (sigmoid). Lets start out by implementing a multiply gate. I'm using Javascript here which has a funny way of simulating classes using functions. If you're not a Javascript - familiar person, all that's going on here is that I'm defining a class that has certain properties (accessed with use of `this` keyword), and some methods (which in Javascript are placed into the function's *prototype*). Just think about these as class methods. Also keep in mind that the way we will use these eventually is that we will first `forward` all the gates one by one, and then `backward` all the gates in reverse order. Here is the implementation:

```javascript

var multiplyGate = function(){ };
multiplyGate.prototype = {
  forward: function(u0, u1) {
    this.u0 = u0; // store pointers to input units
    this.u1 = u1; 
    this.utop = new Unit(u0.value * u1.value, 0.0);
    return this.utop;
  },
  backward: function() {
    this.u0.grad += this.u1.value * this.utop.grad;
    this.u1.grad += this.u0.value * this.utop.grad;
  }
}
```

The multiply gate takes two units that each hold a value and creates a unit that stores its output. The gradient is initialized to zero. Then notice that in the `backward` function call we get the gradient from the output unit we produced during the forward pass (which will by now hopefully have its gradient filled in) and multiply it with the local gradient for this gate (chain rule!). This gate computes multiplication (`u0.value * u1.value`) during forward pass, so recall that the gradient w.r.t `u0` is `u1.value` and w.r.t `u1` is `u0.value`. Also note that we are using `+=` to add onto the gradient in the `backward` function. This will allow us to possibly use the output of one gate multiple times (think of it as a branching wire), since it turns out that the gradients from these different branches just add up when computing the final gradient with respect to the circuit output. The other two gates are defined analogously:

```javascript
var addGate = function(){ };
addGate.prototype = {
  forward: function(u0, u1) {
    this.u0 = u0; 
    this.u1 = u1; // store pointers to input units
    this.utop = new Unit(u0.value + u1.value, 0.0);
    return this.utop;
  },
  backward: function() {
    // add gate. derivative wrt both inputs is 1
    this.u0.grad += 1 * this.utop.grad;
    this.u1.grad += 1 * this.utop.grad;
  }
}

var sigmoidGate = function() { 
  // helper function
  this.sig = function(x) { return 1 / (1 + Math.exp(-x)); };
};
sigmoidGate.prototype = {
  forward: function(u0) {
    this.u0 = u0;
    this.utop = new Unit(this.sig(this.u0.value), 0.0);
    return this.utop;
  },
  backward: function() {
    var s = this.sig(this.u0.value);
    this.u0.grad += (s * (1 - s)) * this.utop.grad;
  }
}
```

Note that, again, the `backward` function in all cases just computes the local derivative with respect to its input and then multiplies on the gradient from the unit above (i.e. chain rule). To fully specify everything lets finally write out the forward and backward flow for our 2-dimensional neuron with some example values:

```javascript
// create input units
var a = new Unit(1.0, 0.0);
var b = new Unit(2.0, 0.0);
var c = new Unit(-3.0, 0.0);
var x = new Unit(-1.0, 0.0);
var y = new Unit(3.0, 0.0);

// create the gates
var mulg0 = new multiplyGate();
var mulg1 = new multiplyGate();
var addg0 = new addGate();
var addg1 = new addGate();
var sg0 = new sigmoidGate();

// do the forward pass
var forwardNeuron = function() {
  ax = mulg0.forward(a, x); // a*x = -1
  by = mulg1.forward(b, y); // b*y = 6
  axpby = addg0.forward(ax, by); // a*x + b*y = 5
  axpbypc = addg1.forward(axpby, c); // a*x + b*y + c = 2
  s = sg0.forward(axpbypc); // sig(a*x + b*y + c) = 0.8808
};
forwardNeuron();

console.log('circuit output: ' + s.value); // prints 0.8808
```

And now lets compute the gradient: Simply iterate in reverse order and call the `backward` function! Remeber that we stored the pointers to the units when we did the forward pass, so every gate has access to its inputs and also the output unit it previously produced.

```javascript
s.grad = 1.0;
sg0.backward(); // writes gradient into axpbypc
addg1.backward(); // writes gradients into axpby and c
addg0.backward(); // writes gradients into ax and by
mulg1.backward(); // writes gradients into b and y
mulg0.backward(); // writes gradients into a and x
```

Note that the first line sets the gradient at the output (very last unit) to be `1.0` to start off the gradient chain. This can be interpreted as tugging on the last gate with a force of `+1`. In other words, we are pulling on the entire circuit to induce the forces that will increase the output value. If we did not set this to 1, all gradients would be computed as zero due to the multiplications in the chain rule. Finally, lets make the inputs respond to the computed gradients and check that the function increased:

```javascript
var step_size = 0.01;
a.value += step_size * a.grad; // a.grad is -0.105
b.value += step_size * b.grad; // b.grad is 0.315
c.value += step_size * c.grad; // c.grad is 0.105
x.value += step_size * x.grad; // x.grad is 0.105
y.value += step_size * y.grad; // y.grad is 0.210

forwardNeuron();
console.log('circuit output after one backprop: ' + s.value); // prints 0.8825
```

Success! `0.8825` is higher than the previous value, `0.8808`. Finally, lets verify that we implemented the backpropagation correctly by checking the numerical gradient:

```javascript
var forwardCircuitFast = function(a,b,c,x,y) { 
  return 1/(1 + Math.exp( - (a*x + b*y + c))); 
};
var a = 1, b = 2, c = -3, x = -1, y = 3;
var h = 0.0001;
var a_grad = (forwardCircuitFast(a+h,b,c,x,y) - forwardCircuitFast(a,b,c,x,y))/h;
var b_grad = (forwardCircuitFast(a,b+h,c,x,y) - forwardCircuitFast(a,b,c,x,y))/h;
var c_grad = (forwardCircuitFast(a,b,c+h,x,y) - forwardCircuitFast(a,b,c,x,y))/h;
var x_grad = (forwardCircuitFast(a,b,c,x+h,y) - forwardCircuitFast(a,b,c,x,y))/h;
var y_grad = (forwardCircuitFast(a,b,c,x,y+h) - forwardCircuitFast(a,b,c,x,y))/h;
```

Indeed, these all give the same values as the backpropagated gradients `(-0.105, 0.315, 0.105, 0.105, 0.210)`. Nice! 

I hope it is clear that even though we only looked at an example of a single neuron, the code I gave above generalizes in a very straight-forward way to compute gradients of arbitrary expressions (including very deep expressions #foreshadowing). All you have to do is write small gates that compute local, simple derivatives w.r.t their inputs, wire it up in a graph, do a forward pass to compute the output value and then a backward pass that chains the gradients all the way to the input. In other words, you can use the code above to learn entire neural networks made up of as many neurons as you like, this is all there's to it! (Of course, in practice we'll want to take efficiency shortcuts. We'll look at these later).

> "You can use that code to learn entire neural networks, this is all there's to it!"

Speaking of neural networks with multiple neurons, now that we have a good intuitive grasp of backpropagation lets venture into the land of Machine Learning, start talking about datasets and learning parameters for deep classification/regression models!

## Chapter 2: Machine Learning

In the previous chapter we saw that we can feed some input through arbitrarily complex real-valued circuit, tug at the end of the circuit with some force, and backpropagation distributes that tug through the entire circuit all the way back to the inputs. If the inputs respond slightly along the final direction of their tug, the circuit will "give" a bit along the original pull direction. Maybe this is not immediately obvious, but this machinery is a powerful *hammer* for Machine Learning.

> "Maybe this is not immediately obvious, but this machinery is a powerful *hammer* for Machine Learning."

### Binary Classification

As we did before, lets start out simple. The simplest, common and yet very practical problem in Machine Learning is **binary classification**. A lot of very interesting and important problems can be reduced to it. The setup is as follows: We are given a dataset of `N` vectors and every one of them is labeled with a `+1` or a `-1`. For example, in two dimensions our dataset could look as simple as:

```
vector -> label
---------------
[1.2, 0.7] -> +1
[-0.3, 0.5] -> -1
[-3, -1] -> +1
[0.1, 1.0] -> -1
[3.0, 1.1] -> -1
[2.1, -3] -> +1
```

Here, we have `N = 6` **datapoints**, where every datapoint has two **features**. Three of the datapoints have **label** `+1` and the other three label `-1`. This is a silly toy example, but in practice a 1/-1 dataset could be very useful things indeed. For example spam/no spam emails, where the vectors somehow measure various features of the content of the email, such as the number of times certain ... enhancement drugs are mentioned.

Our **goal** in binary classification is to learn a parameterized function that takes a 2-dimensional vector and predicts the label. We will want to tune the parameters of the function so that its outputs are consistent with the labeling in the provided dataset. In the end we can discard the dataset and use the learned function as a label predictor for previously unseen vectors.

#### Training protocol

We will eventually build up to entire neural networks and complex expressions, but lets start out simple and train a linear classifier very similar to the single neuron we saw at the end of Chapter 1. The only difference is that we'll get rid of the sigmoid because it makes things unnecessarily complicated (I only used it as an example in Chapter 1 because sigmoid neurons are historically popular but modern Neural Networks rarely use sigmoid neurons). Anyway, lets just use a simple linear function:

$$
f(x, y) = ax + by + c
$$

In this expression we think of `x` and `y` as the inputs (the 2-D vectors) and `a,b,c` as the parameters of the function that we will want to learn. For example, if `a = 1, b = -2, c = -1`, then the function will take the first datapoint (`[1.2, 0.7]`) and output `1 * 1.2 + (-2) * 0.7 + (-1) = -1.2`. Here is how the training will work:

1. We select a random datapoint and feed it through the circuit
2. We will interpret the output of the circuit as a confidence that the datapoint has class `+1`. (i.e. very high values = circuit is very certain datapoint has class `+1` and very low values = circuit is certain this datapoint has class `-1`.)
3. We will measure how well the prediction aligns with the provided labels. Intuitively, for example, if a positive example scores very low, we will want to tug in the positive direction on the circuit, demanding that it should output higher value for this datapoint. Note that this is the case for the the first datapoint: it is labeled as `+1` but our predictor unction only assigns it value `-1.2`. We will therefore tug on the circuit; We want the value to be higher.
4. The circuit will take the tug and backpropagate it to compute tugs on the inputs `a,b,c,x,y`
5. Since we think of `x,y` as (fixed) datapoints, we will ignore the pull on `x,y`. If you're a fan of my physical analogies, think of these inputs as pegs, fixed in the ground.
6. On the other hand, we will take the parameters `a,b,c` and make them respond to their tug (i.e. we'll perform what we call a **parameter update**). This, of course, will make it so that the circuit will output a slightly higher score on this particular datapoint in the future.
7. Iterate! Go back to step 1.

The training scheme I described above, by the way, is commonly referred as **Stochastic Gradient Descent**. The interesting part I'd like to reiterate is that `a,b,c,x,y` are all made up of the same *stuff* as far as the circuit is concerned: They are inputs to the circuit and the circuit will tug on all of them in some direction. It doesn't know the difference between parameters and datapoints. However, after the backward pass is complete we ignore all tugs on the datapoints (`x,y`) and keep swapping them in and out as we iterate over examples in the dataset. On the other hand, we keep the parameters (`a,b,c`) around and keep tugging on them every time we sample a datapoint. Over time, the pulls on these parameters will tune these values in such a way that the function outputs high scores for positive examples and low scores for negative examples.

#### Learning a Support Vector Machine

As a concrete example, lets learn a **Support Vector Machine**. The SVM is a very popular linear classifier; Its functional form is exactly as I've described in previous section, \\( f(x,y) = ax + by + c\\). At this point, if you've seen an explanation of SVMs you're probably expecting me to define the SVM loss function and see me fumble around trying to explain slack variables, large margins, kernels, duality, etc. But I'd like to instead first describe the *force specification* (I just made this term up by the way) of a Support Vector Machine, which I find much more intuitive. Here it is:

- If we feed a positive datapoint through the SVM circuit and the output value is less than 1, pull on the circuit with force `+1`. This is a positive example so we want the score to be higher for it.
- Conversely, if we feed a negative datapoint through the SVM and the output is greater than -1, then the circuit is giving this datapoint dangerously high score: Pull on the circuit downwards with force `-1`.
- In addition to the pulls above, always add a small amount of pull on the parameters `a,b` (notice, not on `c`!) that pulls them towards zero. We will make this pull proprotional to the value of each of `a,b`. For example, if `a` becomes very high it will experience a strong pull of magnitude `|a|` back towards zero. This pull is something we call **regularization**, and it ensures that neither of our parameters `a` or `b` gets disproportionally large. This would be undesirable because both `a,b` get multiplied to the input features `x,y` (remember the equation is `a*x + b*y + c`), so if either of them is too high, our classifier would be overly sensitive to these features. This isn't a nice property because features can often be noisy in practice, so we awnt our classifier to change relatively smoothly if they wiggle around.

Okay there's been too much text. Lets write the SVM code and take advantage of the circuit machinery we have from Chapter 1:

```javascript
// A circuit: it takes 5 Units (x,y,a,b,c) and outputs a single Unit
// It can also compute the gradient w.r.t. its inputs
var Circuit = function() {
  // create some gates
  this.mulg0 = new multiplyGate();
  this.mulg1 = new multiplyGate();
  this.addg0 = new addGate();
  this.addg1 = new addGate();
};
Circuit.prototype = {
  forward: function(x,y,a,b,c) {
    this.ax = this.mulg0.forward(a, x); // a*x
    this.by = this.mulg1.forward(b, y); // b*y
    this.axpby = this.addg0.forward(this.ax, this.by); // a*x + b*y
    this.axpbypc = this.addg1.forward(this.axpby, c); // a*x + b*y + c
    return this.axpbypc;
  },
  backward: function(gradient_top) { // takes pull from above
    this.axpbypc.grad = gradient_top;
    this.addg1.backward(); // sets gradient in axpby and c
    this.addg0.backward(); // sets gradient in ax and by
    this.mulg1.backward(); // sets gradient in b and y
    this.mulg0.backward(); // sets gradient in a and x
  }
}
```

That's a circuit that simply computes `a*x + b*y + c` and can also compute the gradient. It uses the gates code we developed in Chapter 1. Now lets write the SVM, which doesn't care about the actual circuit. It is only concerned with the values that come out of it, and it pulls on the circuit.

```javascript
// SVM class
var SVM = function() {
  
  // random initial parameter values
  this.a = new Unit(1.0, 0.0); 
  this.b = new Unit(-2.0, 0.0);
  this.c = new Unit(-1.0, 0.0);

  this.circuit = new Circuit();
};
SVM.prototype = {
  forward: function(x, y) { // assume x and y are Units
    this.unit_out = this.circuit.forward(x, y, this.a, this.b, this.c);
    return this.unit_out;
  },
  backward: function(label) { // label is +1 or -1

    // reset pulls on a,b,c
    this.a.grad = 0.0; 
    this.b.grad = 0.0; 
    this.c.grad = 0.0;

    // compute the pull based on what the circuit output was
    var pull = 0.0;
    if(label === 1 && this.unit_out.value < 1) { 
      pull = 1; // the score was too low: pull up
    }
    if(label === -1 && this.unit_out.value > -1) {
      pull = -1; // the score was too high for a positive example, pull down
    }
    this.circuit.backward(pull); // writes gradient into x,y,a,b,c
    
    // add regularization pull for parameters: towards zero and proportional to value
    this.a.grad += -this.a.value;
    this.b.grad += -this.b.value;
  },
  learnFrom: function(x, y, label) {
    this.forward(x, y); // forward pass (set .value in all Units)
    this.backward(label); // backward pass (set .grad in all Units)
    this.parameterUpdate(); // parameters respond to tug
  },
  parameterUpdate: function() {
    var step_size = 0.01;
    this.a.value += step_size * this.a.grad;
    this.b.value += step_size * this.b.grad;
    this.c.value += step_size * this.c.grad;
  }
};
```

Now lets train the SVM with Stochastic Gradient Descent:

```javascript
var data = []; var labels = [];
data.push([1.2, 0.7]); labels.push(1);
data.push([-0.3, -0.5]); labels.push(-1);
data.push([3.0, 0.1]); labels.push(1);
data.push([-0.1, -1.0]); labels.push(-1);
data.push([-1.0, 1.1]); labels.push(-1);
data.push([2.1, -3]); labels.push(1);
var svm = new SVM();

// a function that computes the classification accuracy
var evalTrainingAccuracy = function() {
  var num_correct = 0;
  for(var i = 0; i < data.length; i++) {
    var x = new Unit(data[i][0], 0.0);
    var y = new Unit(data[i][1], 0.0);
    var true_label = labels[i];

    // see if the prediction matches the provided label
    var predicted_label = svm.forward(x, y).value > 0 ? 1 : -1;
    if(predicted_label === true_label) {
      num_correct++;
    }
  }
  return num_correct / data.length;
};

// the learning loop
for(var iter = 0; iter < 400; iter++) {
  // pick a random data point
  var i = Math.floor(Math.random() * data.length);
  var x = new Unit(data[i][0], 0.0);
  var y = new Unit(data[i][1], 0.0);
  var label = labels[i];
  svm.learnFrom(x, y, label);

  if(iter % 25 == 0) { // every 10 iterations... 
    console.log('training accuracy at iter ' + iter + ': ' + evalTrainingAccuracy());
  }
}
```
This code prints the following output:

```
training accuracy at iteration 0: 0.3333333333333333
training accuracy at iteration 25: 0.3333333333333333
training accuracy at iteration 50: 0.5
training accuracy at iteration 75: 0.5
training accuracy at iteration 100: 0.3333333333333333
training accuracy at iteration 125: 0.5
training accuracy at iteration 150: 0.5
training accuracy at iteration 175: 0.5
training accuracy at iteration 200: 0.5
training accuracy at iteration 225: 0.6666666666666666
training accuracy at iteration 250: 0.6666666666666666
training accuracy at iteration 275: 0.8333333333333334
training accuracy at iteration 300: 1
training accuracy at iteration 325: 1
training accuracy at iteration 350: 1
training accuracy at iteration 375: 1 
```

We see that initially our classifier only had 33% training accuracy, but by the end all training examples are correctly classifier as the parameters `a,b,c` adjusted their values according to the pulls we exerted. We just trained an SVM! But please don't use this anywhere in production :)

One thing I'd like you to appreciate is that the circuit can be arbitrary expression, not just the linear prediction function we used in this example. For example, it can be an entire neural network.

By the way, I intentionally structured the code in a modular way, but we could have trained an SVM with a much simpler code. Here is really what all of these classes and computations boil down to:

```javascript
var a = 1, b = -2, c = -1; // initial parameters
for(var iter = 0; iter < 400; iter++) {
  // pick a random data point
  var i = Math.floor(Math.random() * data.length);
  var x = data[i][0];
  var y = data[i][1];
  var label = labels[i];

  // compute pull
  var score = a*x + b*y + c;
  var pull = 0.0;
  if(label === 1 && score < 1) pull = 1;
  if(label === -1 && score > -1) pull = -1;

  // compute gradient and update parameters
  var step_size = 0.01;
  a += step_size * (x * pull - a);
  b += step_size * (y * pull - b);
  c += step_size * (1 * pull);
}
```

this code gives an identical result. Perhaps by now you can glance at the code and see what all the equations are doing.

#### A more Conventional Approach: Loss Functions

SVMs and many other Machine Learning models are usually defined in terms of loss functions, not in terms of a "force specification". However, the force specification is the direct result of a loss function since it is the gradient of the loss. Let me clarify with the SVM example.

todo...


### Regression

### 2-layer Neural Network

## Chapter 3: Backprop in Practice

### Backprop: For-loop style

### Backprop: Vectorized Implementations


