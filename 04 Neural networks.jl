import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()

# <br /><br />
#
# ## Neural networks
#
# Now that we know what neurons are, we are ready for the final step:
# the neural network!. A neural network is literally made out of a network of
# neurons that are connected together.
#
# So far, we have just looked at single neurons, that only have a single output.
# What if we want multiple outputs?
#
#
# ### Multiple output models
#
# Let's now distinguish between apples, bananas, *and* grapes.

W = [10 1;
     20 2;
     30 3]
x = [3;
     2]
W*x

# It takes each column of weights and does the dot product against $x$
# (remember, that's how $\sigma^{(i)}$ was defined) and spits out a vector
# from doing that with each column. The result is a vector, which makes this
# version of the function give a vector of outputs which we can use to encode
# larger set of choices.
#
# Matrix multiplication is also great since **GPUs (Graphics Processing Units,
# i.e. graphics cards) are great matrix multiplication machines**, which means
# that by writing the equation this way, the result can be calculated really fast.


#-

# First step: load the data.

using CSV, DataFrames, Flux, Plots
## Load apple data in CSV.read for each file
apples1 = DataFrame(CSV.File(datapath("data/Apple_Golden_1.dat"), delim='\t', allowmissing=:none, normalizenames=true))
apples2 = DataFrame(CSV.File(datapath("data/Apple_Golden_2.dat"), delim='\t', allowmissing=:none, normalizenames=true))
apples3 = DataFrame(CSV.File(datapath("data/Apple_Golden_3.dat"), delim='\t', allowmissing=:none, normalizenames=true))
## And then concatenate them all together
apples = vcat(apples1, apples2, apples3)
bananas = DataFrame(CSV.File(datapath("data/Banana.dat"), delim='\t', allowmissing=:none, normalizenames=true))
grapes1 = DataFrame(CSV.File(datapath("data/Grape_White.dat"), delim='\t', allowmissing=:none, normalizenames=true))
grapes2 = DataFrame(CSV.File(datapath("data/Grape_White_2.dat"), delim='\t', allowmissing=:none, normalizenames=true))
grapes = vcat(grapes1, grapes2)

#-

## Extract out the features and construct the corresponding labels
x_apples  = [ [apples[i, :red], apples[i, :blue]] for i in 1:size(apples, 1) ]
x_bananas  = [ [bananas[i, :red], bananas[i, :blue]] for i in 1:size(bananas, 1) ]
x_grapes = [ [grapes[i, :red], grapes[i, :blue]] for i in 1:size(grapes, 1) ]
xs = vcat(x_apples, x_bananas, x_grapes)
ys = vcat(fill([1,0,0], size(x_apples)),
          fill([0,1,0], size(x_bananas)),
          fill([0,0,1], size(x_grapes)))

# ### One-hot vectors

# `Flux.jl` provides an efficient representation for one-hot vectors, using
# advanced features of Julia so that it does not actually store these vectors,
# which would be a waste of memory; instead `Flux` just records in which
# position the non-zero element is. To us, however, it looks like all the
# information is being stored:

using Flux: onehot

onehot(2, 1:3)

#-

ys = vcat(fill(onehot(1, 1:3), size(x_apples)),
          fill(onehot(2, 1:3), size(x_bananas)),
          fill(onehot(3, 1:3), size(x_grapes)))

# ## The core algorithm from the previous lecture

## model = Dense(2, 1, σ)
## L(x,y) = Flux.mse(model(x), y)
## opt = SGD(params(model))
## Flux.train!(L, zip(xs, ys), opt)

# ### Visualization

using Plots
plot()

contour!(0:0.01:1, 0:0.01:1, (x,y)->model([x,y]).data[1], levels=[0.5, 0.51], color = cgrad([:blue, :blue]))
contour!(0:0.01:1, 0:0.01:1, (x,y)->model([x,y]).data[2], levels=[0.5,0.51], color = cgrad([:green, :green]))
contour!(0:0.01:1, 0:0.01:1, (x,y)->model([x,y]).data[3], levels=[0.5,0.51], color = cgrad([:red, :red]))

scatter!(first.(x_apples), last.(x_apples), m=:cross, label="apples", color = :blue)
scatter!(first.(x_bananas), last.(x_bananas), m=:circle, label="bananas", color = :green)
scatter!(first.(x_grapes), last.(x_grapes), m=:square, label="grapes", color = :red)
