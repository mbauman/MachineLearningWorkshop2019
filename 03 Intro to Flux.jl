import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()

# # Intro to Flux.jl

# ### Let's load in a whole dataset!
#
# Load the datasets that contain the features of the apple and banana images.
#
# The first task:

using CSV, DataFrames

apples = DataFrame(CSV.File("data/apples.dat", delim='\t', normalizenames=true))
bananas = DataFrame(CSV.File("data/bananas.dat", delim='\t', normalizenames=true))

#-

x_apples  = [ [row.red, row.green] for row in eachrow(apples)]
x_bananas = [ [row.red, row.green] for row in eachrow(bananas)]

# Concatenate the x (features) together to create a vector of all our datapoints, and create the corresponding vector of known labels:

xs = [x_apples; x_bananas]
ys = [fill(0, size(x_apples)); fill(1, size(x_bananas))]

#-

using Flux

model = Dense(2, 1, σ)

# We can evaluate the model (currently initialized with random weights) to see what the output value is for a given input:

model(xs[1])

# And of course we can examine the current loss value for that datapoint:

loss = Flux.mse(model(xs[1]), ys[1])

#-

typeof(loss)

# ### Backpropagation

model.W

#-

model.W.grad

#-

using Flux.Tracker
back!(loss)

#-

model.W.grad

# Now we have all the tools necessary to build a simple gradient descent algorithm!

#-

# ### The easy way
#
# You don't want to manually write out gradient descent algorithms every time!
# Flux, of course, also brings in lots of optimizers that can do this all for you.

@doc Descent

#-

@doc Flux.train!

# So we can simply define our loss function, an optimizer, and then call `train!`. That's basic machine learning with Flux.jl.

model = Dense(2, 1, σ)
L(x,y) = Flux.mse(model(x), y)
opt = Descent()
Flux.train!(L, params(model), zip(xs, ys), opt)

# ## Visualize the result

using Plots
begin
    contour(0:.1:1, 0:.1:1, (x, y) -> model([x,y])[].data, fill=true)
    scatter!(first.(x_apples), last.(x_apples), label="apples")
    scatter!(first.(x_bananas), last.(x_bananas), label="bananas")
    xlabel!("mean red value")
    ylabel!("mean green value")
end
