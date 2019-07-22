import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()

# # Express path to classifying images
#
# This is a quick demo on how to run classification software similar
# to how Google images works.
#
# Julia allows us to load in various pre-trained models for classifying images,
# with the `Metalhead.jl` package.

using Metalhead  # To run type <shift> + enter
using Metalhead: classify

#-

using Images

# Let's download an image of an elephant:

file = download("http://www.mikebirkhead.com/images/EyeForAnElephant.jpg")
image = load(file)

# We'll use the VGG19 model, which is a deep convolutional neural network
# trained on a subset of the ImageNet database.

vgg = VGG19()

# To classify the image using the model, we just run the following command, and
# it returns its best guess at a classification:

# This patches up an old function definition that's out of sync from the downloaded model:
# This will be fixed soon...
@eval Metalhead.Flux.NNlib maxpool(x, dims::Tuple;kws...) = maxpool(x, PoolDims(x,(2,2); kws...))

classify(vgg, image)

# Exercise 1: grab a favorite image, then classify it. Tell us what you got!





# We can do the same with any image we have around, for example Alan's dog, Philip:

image = load("data/philip.jpg")

#-

classify(vgg, image)

# ## What is going on here?

#-

# VGG19 classifies images according to the following 1000 different classes:

Metalhead.ImageNet.imagenet_labels[rand(1:1000,1,1)]

# The model is a Convolutional Neural Network (CNN), made up of a sequence of
# layers of "neurons" with interconnections. The huge number of parameters
# making up these interconnections have previously been learnt to correctly
# predict a set of training images representing each class.

#-

# Running the model on an image spits out the probability that the model assigns to each class:

probs = Metalhead.forward(vgg, image)

# We can now see which are the most likely few labels:

perm = sortperm(probs)
probs[273]

#-

[ Metalhead.ImageNet.imagenet_labels[perm] probs[perm] ][end:-1:end-10, :]

# ## What are the questions to get a successful classifier via machine learning?

#-

# The key questions to obtain a successful classifier in machine learning are:
# - How do we define a suitable model that can model the data adequately?
# - How do we train it on suitably labelled data?
