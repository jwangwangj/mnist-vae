# import Automatic Differentiation 
# You may use Neural Network Framework, but only for building MLPs
# i.e. no fancy probabilistic implementations
using Flux
using MLDatasets
using Statistics
using Logging
using Test
using Random
using StatsFuns: log1pexp
Random.seed!(412414);

#### Probability Stuff
# Make sure you test these against a standard implementation!

# log-pdf of x under Factorized or Diagonal Gaussian N(x|μ,σI)
function factorized_gaussian_log_density(mu, logsig,xs)
  """
  mu and logsig either same size as x in batch or same as whole batch
  returns a 1 x batchsize array of likelihoods
  """
  σ = exp.(logsig)
  return sum((-1/2)*log.(2π*σ.^2) .+ -1/2 * ((xs .- mu).^2)./(σ.^2),dims=1)
end

# log-pdf of x under Bernoulli 
function bernoulli_log_density(logit_means,x)
  """Numerically stable log_likelihood under bernoulli by accepting μ/(1-μ)"""
  b = x .* 2 .- 1 # {0,1} -> {-1,1}
  return - log1pexp.(-b .* logit_means)
end
## This is really bernoulli
@testset "test stable bernoulli" begin
  using Distributions
  x = rand(10,100) .> 0.5
  μ = rand(10)
  logit_μ = log.(μ./(1 .- μ))
  @test logpdf.(Bernoulli.(μ),x) ≈ bernoulli_log_density(logit_μ,x)
  # over i.i.d. batch
  @test sum(logpdf.(Bernoulli.(μ),x),dims=1) ≈ sum(bernoulli_log_density(logit_μ,x),dims=1)
end
  
# sample from Diagonal Gaussian x~N(μ,σI) (hint: use reparameterization trick here)
sample_diag_gaussian(μ,logσ) = (ϵ = randn(size(μ)); μ .+ exp.(logσ).*ϵ)
# sample from Bernoulli (this can just be supplied by library)
sample_bernoulli(θ) = rand.(Bernoulli.(θ))

# Load MNIST data, binarise it, split into train and test sets (10000 each) and partition train into mini-batches of M=100.
# You may use the utilities from A2, or dataloaders provided by a framework
function load_binarized_mnist(train_size=1000, test_size=1000)
  train_x, train_label = MNIST.traindata(1:train_size);
  test_x, test_label = MNIST.testdata(1:test_size);
  @info "Loaded MNIST digits with dimensionality $(size(train_x))"
  train_x = reshape(train_x, 28*28,:)
  test_x = reshape(test_x, 28*28,:)
  @info "Reshaped MNIST digits to vectors, dimensionality $(size(train_x))"
  train_x = train_x .> 0.5; #binarize
  test_x = test_x .> 0.5; #binarize
  @info "Binarized the pixels"
  return (train_x, train_label), (test_x, test_label)
end

function batch_data((x,label)::Tuple, batch_size=100)
  """
  Shuffle both data and image and put into batches
  """
  N = size(x)[end] # number of examples in set
  rand_idx = shuffle(1:N) # randomly shuffle batch elements
  batch_idx = Iterators.partition(rand_idx,batch_size) # split into batches
  batch_x = [x[:,i] for i in batch_idx]
  batch_label = [label[i] for i in batch_idx]
  return zip(batch_x, batch_label)
end
# if you only want to batch xs
batch_x(x::AbstractArray, batch_size=100) = first.(batch_data((x,zeros(size(x)[end])),batch_size))

#==============================================================================#

## Implementing the model

### Load the Data
train_data, test_data = load_binarized_mnist(10000,1000);
# train_data, test_data = load_binarized_mnist();
train_x, train_label = train_data;
test_x, test_label = test_data;

### Test the dimensions of loaded data
@testset "correct dimensions" begin
@test size(train_x) == (784,1000)
@test size(train_label) == (1000,)
@test size(test_x) == (784,1000)
@test size(test_label) == (1000,)
end

#==============================================================================#

## Model Distributions
# log_prior: compute log of the prior over a digit's representation z
log_prior(z) = factorized_gaussian_log_density(0,0,z)

## Model Dimensionality
# Set up model according to Appendix C (using Bernoulli decoder for Binarized MNIST)
# Set latent dimensionality=2 and number of hidden units=500.
Dz, Dh = 2, 500
Ddata = 28^2

## Generative Model
#= 
This will require implementing a simple MLP neural network
See example_flux_model.jl for inspiration
Further, you should read the Basics section of the Flux.jl documentation
https://fluxml.ai/Flux.jl/stable/models/basics/
that goes over the simple functions you will use.
You will see that there's nothing magical going on inside these neural network libraries
and when you implemented a neural network in previous assignments you did most of the work.
If you want more information about how to use the functions from Flux, you can always reference
the internal docs for each function by typing `?` into the REPL:
? Chain
? Dense
=#

### decoder: given a latent representation $z$ produces a 784-dimensional mean 
# vector of a product of Bernoulli distributions, one for each pixel in a 
# $28 \times 28$ image.
# MLP uses a single hidden layer with 500 hidden units and a tanh nonlinearity
#TODO: how to output as logit?
decoder = Chain(Dense(Dz, Dh, tanh), Dense(Dh, Ddata))

### log_likelihood: given a latent representation $z$ and a binarized digit $x$,
# computes the log-likelihood $\log p(x|z)$.
function log_likelihood(x,z)
  """ Compute log likelihood log_p(x|z)"""
  # return likelihood for each element in batch
  p = decoder(z)
  # @info p, x
  return  sum(bernoulli_log_density(p,x),dims=1)
end

### joint_log_density: combines log-prior and log-likelihood of the observations
# to give $\log p(z, x)$ for a single image. 
joint_log_density(x,z) = log_prior(z) .+ log_likelihood(x,z)

#==============================================================================#

## Amortized Inference
function unpack_gaussian_params(θ)
  μ, logσ = θ[1:2,:], θ[3:end,:]
  return  μ, logσ
end

### encoder: evaluates MLP to give mean and log-standard deviation of a 
# factorized Gaussian with Dim D_z=2.
# MLP uses a single hidden layer with 500 hidden units and a tanh nonlinearity
encoder = Chain(Dense(Ddata, Dh, tanh), Dense(Dh, Dz*2), unpack_gaussian_params)
# Hint: last "layer" in Chain can be 'unpack_gaussian_params'

### log_q: write log likelihood under variational distribution.
log_q(q_μ, q_logσ, z) = factorized_gaussian_log_density(q_μ, q_logσ, z)

### elbo: computing unbiased estimate of the elbo on a batch of images xs
function elbo(x)
  # variational parameters from data
  q_μ, q_logσ = encoder(x) 
  @info "$(size(q_μ)) $(size(q_logσ))"
  # sample from variational distribution
  z = sample_diag_gaussian(q_μ, q_logσ) 
  @info "$(size(z))"
  # joint likelihood of z and x under model
  joint_ll = joint_log_density(x,z)
  # likelihood of z under variational distribution 
  log_q_z = log_q(q_μ, q_logσ, z)
  # Scalar value, mean variational evidence lower bound over batch
  elbo_estimate = mean(joint_ll - log_q_z, dims=2)
  return elbo_estimate[1]
end

### loss: gives negative elbo estimate over batch of data x
function loss(x)
  # scalar value for the variational loss over elements in the batch
  return -elbo(x)
end

### train_model_params: initializes and optimizes the encoder and decoder params
# jointly on the training set. Optimizes with gradients on the elbo estimate over 
# batches of data, not the whole dataset. Trains for 10 epochs by default.
# Training with gradient optimization:
# See example_flux_model.jl for inspiration
function train_model_params!(loss, encoder, decoder, train_x, test_x; nepochs=10)
  # model params: parameters to update with gradient descent
  ps = Flux.params(encoder, decoder)
  # ADAM optimizer with default parameters
  opt = ADAM()
  # over batches of the data
  for i in 1:nepochs
    for d in batch_x(train_x)
      # compute gradients with respect to variational loss over batch 
      # first argument is an anonymous function
      gs = Flux.gradient(() -> loss(d), ps) 
      # update the parameters with gradients
      Flux.Optimise.update!(opt, ps, gs)
    end
    if i%1 == 0 # change 1 to higher number to compute and print less frequently
      @info "Test loss at epoch $i: $(loss(batch_x(test_x)[1]))"
    end
  end
  @info "Parameters of encoder and decoder trained!"
end

#==============================================================================#

# ### Train the model
# debug_train=train_x[:, 1:10]
# # train_model_params!(loss,encoder,decoder,train_x,test_x, nepochs=100)
# train_model_params!(loss,encoder,decoder,debug_train,test_x, nepochs=1)

# #### Save the trained model!
# using BSON:@save
# cd(@__DIR__)
# @info "Changed directory to $(@__DIR__)"
# save_dir = "trained_models"
# if !(isdir(save_dir))
#   mkdir(save_dir)
#   @info "Created save directory $save_dir"
# end
# @save joinpath(save_dir,"encoder_params.bson") encoder
# @save joinpath(save_dir,"decoder_params.bson") decoder
# @info "Saved model params in $save_dir"
  
#### Load the trained model!
using BSON:@load
cd(@__DIR__)
@info "Changed directory to $(@__DIR__)"
load_dir = "trained_models"
@load joinpath(load_dir,"encoder_params.bson") encoder
@load joinpath(load_dir,"decoder_params.bson") decoder
@info "Load model params from $load_dir"


#==============================================================================#


## Visualization
using Images
using Plots

### mnist_img: make vector of digits into images, works on batches also
mnist_img(x) = ndims(x)==2 ? Gray.(reshape(x,28,28,:))' : Gray.(reshape(x,28,28))'

### Example for how to use mnist_img to plot digit from training data
# plot(mnist_img(train_x[:,1])) # 5
# plot(mnist_img(train_x[:,2])) # 0
plot(mnist_img(train_x[:,3]))
# plot(mnist_img(train_x[:,4]))
# plot(mnist_img(train_x[:,5]))


### 3a: plot samples from the trained generative model using ancestral sampling
num_samples = 10

# sample a z from the prior.
sample_z = randn((2,num_samples))
# compute the bernoulli means over the pixels of $x$ given $z$ using the generative model.
bernoulli_means = 1.0 ./ (1.0 .+ exp.(-decoder(sample_z)))
# Sample a binary image $x$ from this product of Bernoullis. Plot this sample as an image.
binary_sample = sample_bernoulli(bernoulli_means)

# plots of bernoulli means of p(x|z) for each sample of z
plots_bernoulli = [plot(mnist_img(bernoulli_means[:,i]))  for i in 1:num_samples]
# binary images sampled from the bernoulli distribution
plots_binary =    [plot(mnist_img(binary_sample[:,i]))    for i in 1:num_samples]
# each column is an independent sample
plots = [ plots_bernoulli; plots_binary ]

#### plot and save  3a
display(plot(
  plots..., 
  layout = grid(2,num_samples), 
  size = (125*num_samples, 125*2)
))
savefig(joinpath("plots","A3Q3a.png"))


### 3b scatter plot where each point represents an image in the training set
vector_x = [[] for i in 1:10]
vector_y = [[] for i in 1:10]
for i in 1:size(train_x)[2]
  # encode each image in the trianing set
  q_μ, q_logσ = encoder(train_x[:,i]) 
  # take 2D mean vector of each encoding
  push!(vector_x[1 + train_label[i]], q_μ[1])
  push!(vector_y[1 + train_label[i]], q_μ[2])
end

#### plot and save 3b
display(plot(
  vector_x,
  vector_y,
  seriestype = :scatter,
  title = "Latent Space Encoding from MNIST Training Set",
  xlabel = "z_1",
  ylabel = "z_2",
  label = ["0" "1" "2" "3" "4" "5" "6" "7" "8" "9"],
  size = (800, 800)
))
savefig(joinpath("plots","A3Q3b.png"))


### 3c visualizing generative output along linear interpolations between mean
# vectors of 3 pairs of encoded data points with different classes to examine 
# the latent variable model

# get linear interpolation of 2 points
function linear_interpolation(za, zb, α)
  zα = α .* za + (1 - α) .* zb
  return zα
end 

# sample 3 pairs of images, each with a different class
sample_images = [
  # 5, 0
  (train_x[:,1], train_x[:,2]),
  # 4, 1
  (train_x[:,3], train_x[:,4]),
  # 9, 2
  (train_x[:,5], train_x[:,5]),
]

# encode the data in each pair and take the mean vectors
encoded_sample_images = [
  (encoder(sample_images[i][1]), encoder(sample_images[i][2])) 
  for i in 1:3
]

# make plots of the means at 10 equally spaced points
plots = []
for i in 1:3
  encoded_a, encoded_b = encoded_sample_images[i]
  for j in 1:10
    alpha = j/10.0
    μα = linear_interpolation(encoded_a[1], encoded_b[1], alpha)
    # get Bernoulli means
    logit_μ = decoder(μα)
    # plot means
    ber_μ = exp.(logit_μ) ./ (1 .+ exp.(logit_μ))
    push!(plots, plot(mnist_img(ber_μ[:])))
  end
end

#### plot and save
display(plot(
  plots..., 
  layout=grid(3,10), 
  size =(850, 250),
  # layout=grid(10, 3), 
  # size =(500, 1700),
  axis = nothing
))
savefig(joinpath("plots","A3Q3c.png"))

#==============================================================================#

## Predicting the bottom

### compute p(z, top half of image x)
#### top_half: gets top half of a 28x28 array
function top_half(x)
  half_image = x[1:14*28, :]
  return half_image
end

#### top_half_zeros: gets top half of a 28x28 array and adds zeros
function top_half_zeros(x)
  half_image = x[1:14*28, :]
  z = zeros(Int8, 14*28, size(x)[2])
  return [half_image; z]
end

#### make_square: makes 784 rows into 28x28 squares
function make_square(x)
  img = reshape(x, 28, 28, size(x)[2], :)
  return permutedims(img, [2, 1, 3, 4])
end

#### log_likelihood_top_half: computes log p(top half of image x| z)
function log_likelihood_top_half(top_half_x, z)
  p = decoder(z)
  half_p = top_half(p)
  # sums top half
  sum(bernoulli_log_density(half_p, top_half_x), dims=1)
end

#### joint_log_density_top_half: compute log p(z, top half of image x)
# compute 3 log densities for the elbo: log p(z), log p(x|z), and log q(z).
# log p(z) and log q(z) are both 2-dimensional Gaussians.
joint_log_density_top_half(z, top_half_x) = 
  log_prior(z) .+ log_likelihood_top_half(top_half_x, z)

### approximate $p(z | \text{top half of image x})$ in a scalable way using
# stochastic variational inference.

#### elbo_svi: gets elbo over K samples $z ∼ q(z|top half of x)$
function elbo_svi(params, logp)
  # needs to be (2, k)
  p_μ, p_logσ = params
  # @info "elbo svi: $(size(p_μ)) $(size(p_logσ))"
  # needs to be (2, k)
  z = sample_diag_gaussian(p_μ, p_logσ)
  # @info "elbo svi samples z: $(size(z))"
  joint_ll = logp(z)
  # @info "joint_ll: $(size(joint_ll))"
  # likelihood of z under variational distribution 
  log_q_z = log_q(p_μ, p_logσ, z)
  # return sum(logp_estimate - logq_estimate) / k
  elbo_estimate = mean(joint_ll - log_q_z, dims=2)
  # Scalar value, mean variational evidence lower bound over batch
  # @info "elbo_estimate: $(elbo_estimate[1])"
  return elbo_estimate[1]
end

#### neg_elbo_svi: Takes parameters for q, evidence as an array of game outcomes,
# and returns the -elbo estimate with k many samples from q
function neg_elbo_svi(params; x = train_x) 
  # print(size(x))
  half_x = top_half(x)
  logp(z) = joint_log_density_top_half(z, half_x)
  return -elbo_svi(params, logp)
end

#### contours
function contours!(f; colour=nothing)
  n = 100
  x = range(-1,stop=0.5,length=n)
  y = range(0,stop=1.5,length=n)
  z_grid = Iterators.product(x,y) # meshgrid for contour
  z_grid = reshape.(collect.(z_grid),:,1) # add single batch dim
  z = f.(z_grid)
  z = getindex.(z,1)'
  max_z = maximum(z)
  levels = [.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2] .* max_z
  if colour==nothing
  p1 = contour!(x, y, z, fill=false, levels=levels)
  else
  p1 = contour!(x, y, z, fill=false, c=colour,levels=levels,colorbar=false)
  end
  plot!(p1)
end

### train_model_params_svi: optimize ϕ_μ and ϕ_logσ 
function train_model_params_svi!(init_params, data; num_itrs=200, lr= 1e-2)
  @info "Training model params using SVI"
  params_cur = init_params 
  for i in 1:num_itrs
    # gradients of variational objective with respect to parameters
    grad_params = gradient(
      params -> neg_elbo_svi(params, x = data), params_cur
    )[1]
    # update paramters with lr-sized step in descending gradient
    params_cur =  params_cur .- lr .* grad_params
    # report the current elbo during training
    if i == num_itrs || i%10 == 0 
      @info "Loss at iter $i: $(neg_elbo_svi(params_cur, x = data))"

      pl = plot(title="Train model params using SVI: iteration $i");
      # likelihood contours for target posterior
      contours!(zs -> exp.(joint_log_density_top_half(zs, data)), colour=:red) 
      # likelihood contours for variational posterior
      contours!(
          zs -> factorized_gaussian_log_density(params_cur[1], params_cur[2], zs), 
          colour=:blue
        )
      # true posterior in red and variational in blue
      display(pl)
    elseif i%1 == 0 
        @info "Loss at iter $i: $(neg_elbo_svi(params_cur, x = data))"
    end
  end
  return params_cur
end

### set up training model
function set_up_svi(num_samples)
  train = train_x[:, 1:num_samples]
  half_train = top_half(train)
  ϕ_μ = randn(2, num_samples)
  ϕ_logσ = randn(2, num_samples)
  return half_train, ϕ_μ, ϕ_logσ
end

#### train the model
half_train, ϕ_μ, ϕ_logσ = set_up_svi(5000)
params_svi = train_model_params_svi!((ϕ_μ, ϕ_logσ), half_train; num_itrs=200, lr= 1e-2)
# half_train, ϕ_μ, ϕ_logσ = set_up_svi(1000)
# params_svi_1000 = train_model_params_svi!((ϕ_μ, ϕ_logσ), top_half(train_x); num_itrs=200, lr= 1e-2)

### plotting 

pl = plot(title="Trained model params using SVI");
# likelihood contours for target posterior
contours!(zs -> exp.(joint_log_density_top_half(zs, half_train)), colour=:red) 
# likelihood contours for variational posterior
contours!(
    zs -> factorized_gaussian_log_density(params_svi[1], params_svi[2], zs), 
    colour=:blue
  )
# true posterior in red and variational in blue
display(pl)

### sampling 


#==============================================================================#

#### testing loss
loss(batch_x(train_x)[1])

##### testing top_half
tt = train_x[:, 1:3]
htt = tt[1:14*28, :]
i = reshape(htt, 28, 14, size(htt)[2], :)
permutedims(i, [2, 1, 3, 4])
top_half(tt)

z = zeros(Int8, 14, 28)
o = ones(Int8, 14, 28)
[z; o]

aa = reshape([1 2 3 4 5 6 7 8 9 1 2 3 4 5 6 7 8 9 1 2 3 4 5 6 7 8 9], 9, 3)
a = aa[1:4, :]
a = [a; zeros(Int8, 4, 3)]
size(a)
ha = reshape(a, 4, 2, size(a)[2], :)
size(ha)
permutedims(ha, [2, 1, 3, 4])


#### testing things
t = train_x[:, 1:10]
a = randn(2, 10)
b = randn(2, 10)
params = (a, b)


#### testing neg elbo and elbo
t = train_x[:, 1:10]
a = randn(2, 10)
b = randn(2, 10)

neg_elbo_svi((a, b), x = t)
s = sample_diag_gaussian(a, b)
ht = top_half(t)

joint_log_density_top_half(s, ht)
log_prior(s)
log_likelihood_top_half(ht, s)
m = decoder(s)
hm = top_half(m)
bernoulli_log_density(hm,ht)

mean(joint_log_density_top_half(s, ht) - log_q(a, b, s), dims=2)



#### testing things
a, b = encoder(train)
size(a)
size(b)
p = randn(4, 500)
a, b = unpack_gaussian_params(p)
decoder(a)

