# ## What is learning?
#
# We want our machine to *learn* the parameters in that black box, based upon the data

#-

# ### Motivation: Fitting parameters by hand
#

using Plots, Images, Statistics

sigma(x,w,b) = 1 / (1 + exp(-w*x+b))

apple =  load("data/10_100.jpg")
banana = load("data/104_100.jpg")
apple_green_amount =  mean(Float64.(green.(apple)))
banana_green_amount = mean(Float64.(green.(banana)))

begin
    w = 10 # Change these values to move the line to be "representative"
    b = 10

    plot(x->sigma(x,w,b), 0, 1, label="Model", legend = :topleft, lw=3)
    scatter!([apple_green_amount],  [0.0], label="Apple")
    scatter!([banana_green_amount], [1.0], label="Banana")

end

# Intuitively, how did you tweak those values so that way the model sends
# apples to 0 and bananas to 1?

#-

# ## "Learning by nudging": The process of descent
#
    # Let's start to formalize this idea. In order to push the curve in the "right direction", we need some measurement of "how right" and "how wrong" the model is. When we translate the idea of a "right direction" into math, we end up with a **loss function**, `L(w, b)`, as we saw in notebook 5. We say that the loss function is lowest when the model `sigma(x, w, b)` performs the best.
#
# Now we want to create a loss function that is the lowest when the apple is at `0` and the banana is at `1`. If the data (the amount of green) for our apple is $x_1$, then our model will output $sigma(x_1,w, b)$ for our apple. So, we want the difference $0 - sigma(x_1, w, b)$ to be small. Similarly, if our data for our banana (the banana's amount of green) is $x_2$, we want the difference $1 - sigma(x_2, w, b)$ to be small.
#
# To create our loss function, let's add together the squares of the difference of the model's output from the desired output for the apple and the banana. We get
#
# $$ L(w,b) = (0 - sigma(x_1, w, b))^2 + (1 - sigma(x_2, w, b))^2. $$
#
# $L(w, b)$ is lowest when it outputs `0` for the apple and `1` for the banana, and thus the cost is lowest when the model "is correct".


L(w, b) = (0 - sigma(apple_green_amount,w,b))^2 + (1 - sigma(banana_green_amount,w,b))^2

w_range = 10:0.1:13
b_range = 0:1:20

L_values = [L(w,b) for b in b_range, w in w_range]


begin
    w = 10 # Use values between 10 and 13
    b = 10 # Use values between 0 and 20
    p1 = contour(w_range, b_range, L_values, levels=0.05:0.1:1, xlabel="w", ylabel="b", cam=(70,40), cbar=false, leg=false)
    scatter!(p1, [w], [b], markersize=5, color = :blue)

    p2 = plot(x->sigma(x,w,b), 0, 1, label="Model", legend = :topleft, lw=3)
    scatter!(p2, [apple_green_amount],  [0.0], label="Apple", markersize=10)
    scatter!(p2, [banana_green_amount], [1.0], label="Banana", markersize=10, xlim=(0,1), ylim=(0,1))
    plot(p1, p2, layout=(2,1))
end

# The blue ball on the 3D plot shows the current parameter choices, plotted as `(w,b)`. Shown below the 3D plot is a 2D plot of the corresponding model with those parameters. Notice that as the blue ball rolls down the hill, the model becomes a better fit. Our loss function gives us a mathematical notion of a "hill", and the process of "learning by nudging" is simply rolling the ball down that hill.

#-

# To do this mathematically, we need to know which direction is "downhill". Recall from calculus that the derivative of `L` with respect to `b` tells you how `L` changes when `b` changes. Thus to roll downhill, we should go in the direction where the derivative is negative (the function goes down) for each parameter. This direction is the negative of what's called the **gradient**, $\nabla L$. This means that the "learn by nudging method" can be rephrased in mathematical terms as:
#
# 1. Calculate the gradient
# 2. Move a little bit in the direction of the negative gradient
# 3. Repeat
#
# This process of rolling the ball in the direction of the negative gradient is
# called **gradient descent**.
#
# If we repeat this process, then we will end up at parameters where the model
# correctly labels apples as `0` and bananas as `1`. When this happens, the
# model has learned from the data and can then read pictures and tell you whether they are apples or bananas!

# The real trick is doing it with lots of data — and not just two examples.
