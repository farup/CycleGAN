# CycleGAN
Project from the course DeepLearning at the Free University of Bozen-Bolzano

## Table of Conent 
- [Introduction](#introduction)
- [CNN – Brief Recap](#cnn–brief-recap)
- [Generative Models](#generative-models)
- [CycleGANs](#cyclegans)
- [Adversarial Loss](#adversarial-loss)
- [Cycle Consistency Loss](#cycle-consistency-loss)
- [Project Implementation](#project-implementation)
- [Code](#code)
- [Results](#results)

## Introduction 
This a implementation of the paper ”Unpaired Imageto-Image Translation using Cycle-Consistent Adversarial Networks”, Paper by Jun-Yan Zhu,Taesung Park Phillip and Isola Alexei A. Efros. Following content is a summary of my related report. 

The goal Is to learn a “translation” G: X -> Y and F: Y -> X given two unordered image collections X and Y, in the absence of paired examples. In this case the purpose is to “translate” images how horses (X) to images of zebras (Y), and vice versa.This is done by playing an adversial game between generators and discriminators.  The goal of the generators, G, is to produce samples from a distribution, while the discriminators aim to figure out if the sample is from an actual distribution (real) or generated (fake). More precisely, we want to optimize adversarial- and cycle consistency loss. 

<a href="https://arxiv.org/pdf/1703.10593v7.pdf">Link to paper</a>

<a href="https://www.kaggle.com/datasets/suyashdamle/cyclegan">Link to datasets</a>

## Theory 

### CNN – Brief Recap 
CNN is based on the shared-weight architecture of kernels or filters which slide along the input features. The type of operation applied between the filter-sized patch on the input features and the filter is a dot product, which returns a single value in an output-map, known as a feature map. Each value of the feature map is then usually passed through a non-linearity function such as a ReLU. 

The clever invention of CNN is to learn filter weights during the training of the network. As usually, we want to minimize the loss function. This forces the network to extract features we are looking for, which requires as specific set of filter weights. By calculating the gradient of the loss function and propagating this error backward, we can determine the gradient descent together with a learning rate. This tunes the weights in the direction that minimizes the loss.

### Generative Models 
Previously much of the advances in machine learning has been in discriminative models where we try to estimate the posterior probability. This is the probability p(y|x). Linear regression falls in this class, but also deep neural classifiers where x represents the image and y the label.
Generative models instead assumes that any sample of data is generated from a distribution and tries to estimate this distribution. Once the distribution is estimated the model could be used to generate samples following this distribution.  The estimation of the distribution is done because learning the true probability distribution p(x) is infeasible in finite time. For example, learning the probability distribution of images over horses we need to define a model which can model complex correlations between all pixels between in each image.

Instead of modelling the marginal p(x) directly, we introduce unobserved latent variable z. Latent variables are a transformation of the data points into a continuous lower-dimensional space. Intuitively, the latent variables will describe or “explain” the data in a simpler way. For the example of images of horses, z would represent a vector where each dimension represents some attribute about the data,  eyes, color … With this we can define a conditional distribution p(x|z), likelihood of data/observations given latent variable for the data. 

Having z, we can further introduce a prior distribution p(z) over the latent variables to compute the joint distribution over observed and latent variables.

This allows us to express the complex distribution p(x) in its components p(x|z) and p(z). 

p(x) = ∫p(x,z)dz =∫p(x|z)p(z)dz

To obtain p(x) we need to marginalize of the latent variables. 

In most cases the equation above does not have an analytical solution, and we in deeplearning models we will approximate the likelihood distribution  p(x|z) with a neural network.

### Generative Adversarial Network - GANs
Where Variational Autoencoders VAE tries to model the encoded latent vector in a probabilistic matter, p(z|x), GANs give up on this. Instead, it sample from a gaussian distributed z space, and a following adversarial process in which two models “players” are trained simultaneously, also referred to as wo player game. The two players; 

-	The generator G that learns to generate plausible data from the gaussian distributed noise.
-	The discriminator D that learns to distinguish the generator’s fake data from real data. 

$$min_Gmax_D V(G,D) = E_{x \sim p_{data}} [ln(D(x))] + E_{z \sim p_{z}}[ln(1-D(G(z))]$$

### CycleGANs 

As we know from previously, CycleGAN wants to learn the two mappings G : X− > Y and F : Y − > X. Two adversarial discriminators Dx and Dy are therefore introduced. Dx aims to distinguish between images x and translated images F(y), while Dy aims to discriminate between y and G(x). In addition to two adversarial loss, cycle consistency loss is also introduced.

### Adversarial Loss

For the mapping function G and its discriminator Dy the objective is:

$$L_{GAN}(G,D_y, X,Y) = E_{y\sim p_{data(y)}} [ln(D_y(y))] + E_{x \sim p_{data(x)}}[ln(1-D(G(x))]$$

where G tries to generate images G(x) that look like images from domain Y, while Dy aims to distinguish between G(x) and real samples y. Dy tries to maximise this objective as it wants to Dy(y) to output 1 (real) and Dy(G(x)) output 0 (fake). The adversary G aims the to minimize this objective. Similar adversarial loss for the mapping F and discriminator Dx

### Cycle Consistency Loss

With large enough capacity, a network can map the same set of input images to any random style of the images in the target domain. Thus adversarial losses alone cannot guarantee that the learned function can map an individual input xi to a desired output yi. As previously mentioned, cycle consistency is therefore implemented

$$ x -> G(x) -> F(G(x)) \sim x$$

Cycle consistency 

$$L_{cyc}(G,F) = E_{x\sim p_{data(x)}} [|| F(G(X)) - X||_1] + E_{y \sim p_{data(y)}}[G(F(y)) - y||_1]$$

Cycle consistency loss.

## Project Implementation 

Dataset in this project is ”horse2zebra”. The dataset consist of 256 x 256 pixel images, with 1067 train samples of horses and 1334 zebras. Aswell as test for both horses and zebras. No data reprocessing was applied.

### Code 

Project follows the network architecture described in the related papers 7.Appendix. The blocks are implemented in terms of functions returning nn.Sequential(), with goal of reusable and easy modifications. Discriminator and Generator are implemented as two separate classes. The dataset class ”HorseZebraDataset” loads the dataset from a under a folder ’/dataset/’ uploaded on google drive. Due to few parameters details in the architecture descriptions, I adjusted strides and padding, such that the tensors feature maps to matched. The number of channels are according to the architecture description. Paper suggest to replaces the negative log likelihood objective (loss function) by the least-squares loss, due to more stable loss. Adam solver is used as optimizer, with batch size of 1. The discriminators and generators are trained simultaneously. Scaling of gradients are also performed, so they aren’t flushed to zero.

## Results 

The images show results after training the network through 10 epochs. As
we can see, the results are rather interesting. Future work will look more into the data preprocessing/transformations and selected parameters. 
