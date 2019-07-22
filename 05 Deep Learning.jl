import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()

# <br />
# ## Going deep: Deep neural networks
#
# So far, we've learned that if we want to classify more than two fruits, we'll
# need to go beyond using a single neuron and use *multiple* neurons to get
# multiple outputs. We can think of stacking these multiple neurons together in
# a single neural layer.
#
# Even so, we found that using a single neural layer was not enough to fully
# distinguish between bananas, grapes, **and** apples. To do this properly, we'll
# need to add more complexity to our model. We need not just a neural network,
# but a *deep neural network*.
#
# There is one step remaining to build a deep neural network. We have been
# saying that a neural network takes in data and then spits out `0` or `1`
# predictions that together declare what kind of fruit the picture is. However,
# what if we instead put the output of one neural network layer into another
# neural network layer?

# # Deep learning with Flux

#-

# Let's load the same datasets from the previous lecture and pre-process them in the same way:

using CSV, DataFrames, Flux, Plots
apples1 = DataFrame(CSV.File("data/Apple_Golden_1.dat", delim='\t', allowmissing=:none, normalizenames=true))
apples2 = DataFrame(CSV.File("data/Apple_Golden_2.dat", delim='\t', allowmissing=:none, normalizenames=true))
apples3 = DataFrame(CSV.File("data/Apple_Golden_3.dat", delim='\t', allowmissing=:none, normalizenames=true))
apples = vcat(apples1, apples2, apples3)
bananas = DataFrame(CSV.File("data/Banana.dat", delim='\t', allowmissing=:none, normalizenames=true))
grapes1 = DataFrame(CSV.File("data/Grape_White.dat", delim='\t', allowmissing=:none, normalizenames=true))
grapes2 = DataFrame(CSV.File("data/Grape_White_2.dat", delim='\t', allowmissing=:none, normalizenames=true))
grapes = vcat(grapes1, grapes2)
## Extract out the features and construct the corresponding labels
x_apples  = [ [apples[i, :red], apples[i, :blue]] for i in 1:size(apples, 1) ]
x_bananas  = [ [bananas[i, :red], bananas[i, :blue]] for i in 1:size(bananas, 1) ]
x_grapes = [ [grapes[i, :red], grapes[i, :blue]] for i in 1:size(grapes, 1) ]
xs = vcat(x_apples, x_bananas, x_grapes)
ys = vcat(fill(Flux.onehot(1, 1:3), size(x_apples)),
          fill(Flux.onehot(2, 1:3), size(x_bananas)),
          fill(Flux.onehot(3, 1:3), size(x_grapes)));

# In the previous lecture, we used a `Dense(2, 3, σ)` as our model. Now we want to construct multiple layers and chain them together:

layer1 = Dense(2, 4, σ)
layer2 = Dense(4, 3, σ)

#-

layer2(layer1(xs[1]))

#-

#nb ?Chain
#jl @doc Chain

#-

m = Chain(layer1, layer2)
m(xs[1])

#-

xs[1] |> layer1 |> layer2

# ### The core algorithm from the last lecture

model = Chain(Dense(2, 3, σ)) # Update this!
L(x,y) = Flux.mse(model(x), y)
opt = Descent()
Flux.train!(L, params(model), zip(xs, ys), opt)

#-

## Recall his is how we repeatedly walked down our gradient previously...
for _ in 1:1000
    Flux.train!(L, zip(xs, ys), opt)
end
## But our model is now more complicated and this will take more time!

#-

data = zip(xs, ys)
@time Flux.train!(L, data, opt)
@time Flux.train!(L, data, opt)

# ### Improving efficiency by batching

length(data)

#-

first(data)

# Recall our matrix-vector multiplication from the previous lecture:

W = [10 1;
     20 2;
     30 3]
x = [3;
     2]
W*x

#-

Flux.batch(xs)

#-

model(Flux.batch(xs))

#-

databatch = (Flux.batch(xs), Flux.batch(ys))
@time Flux.train!(L, (databatch,), opt)
@time Flux.train!(L, (databatch,), opt)

#-

Flux.train!(L, Iterators.repeated(databatch, 10000), opt)

#-

L(databatch[1], databatch[2])

# ### Visualization

using Plots
function plot_decision_boundaries(model, x_apples, x_bananas, x_grapes)
    plot()

    contour!(0:0.01:1, 0:0.01:1, (x,y)->model([x,y]).data[1], levels=[0.5, 0.501], color = cgrad([:blue, :blue]), colorbar=:none)
    contour!(0:0.01:1, 0:0.01:1, (x,y)->model([x,y]).data[2], levels=[0.5,0.501], color = cgrad([:green, :green]), colorbar=:none)
    contour!(0:0.01:1, 0:0.01:1, (x,y)->model([x,y]).data[3], levels=[0.5,0.501], color = cgrad([:red, :red]), colorbar=:none)

    scatter!(first.(x_apples), last.(x_apples), m=:cross, label="apples", color = :blue)
    scatter!(first.(x_bananas), last.(x_bananas), m=:circle, label="bananas", color = :green)
    scatter!(first.(x_grapes), last.(x_grapes), m=:square, label="grapes", color = :red)
end
plot_decision_boundaries(model, x_apples, x_bananas, x_grapes)

# ### Further improvements with a better loss function and normalization of outputs

scatter([0],[0], label="correct answer", xlabel="model output: [1-x,x]", ylabel="loss against [1, 0]", legend=:topleft, title="Loss function behavior")
plot!(x->Flux.mse([1-x, x/2], [1,0]), -1.5, 1.5, label="mse")
## plot!(x->Flux.crossentropy([1-x, x/2], [1,0]), 0, 1, label="crossentropy")

#-

sum(model(xs[1]))

#-

Flux.mse([0.01,0.98,0.01], [1.0,0,0])

#-

softmax([1.0,-3,0])

# ### The new algorithm
#
# Use `softmax` as a final normalization and change the loss function to `crossentropy`:

model = Chain(Dense(2, 4, σ), Dense(4, 3, identity), softmax)
L(x,y) = Flux.crossentropy(model(x), y)
opt = Descent()

#-

Flux.train!(L, params(model), Iterators.repeated(databatch,5000), opt)

#-

plot_decision_boundaries(model, x_apples, x_bananas, x_grapes)
