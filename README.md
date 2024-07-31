---

# Art Generation with Neural Style Transfer

## Overview

This project demonstrates the implementation of the Neural Style Transfer (NST) algorithm, which merges the content of one image with the style of another to create a novel artistic image. The algorithm, created by Gatys et al. (2015), utilizes a pre-trained convolutional network (VGG-19) to achieve this transfer. The project covers the complete process, from setting up the environment to generating the final stylized image.

## Table of Contents

1. [Packages](#1-packages)
2. [Problem Statement](#2-problem-statement)
3. [Transfer Learning](#3-transfer-learning)
4. [Neural Style Transfer](#4-neural-style-transfer)
   - 4.1 [Computing the Content Cost](#41-computing-the-content-cost)
     - 4.1.1 [Make Generated Image G Match the Content of Image C](#411-make-generated-image-g-match-the-content-of-image-c)
     - 4.1.2 [Content Cost Function](#412-content-cost-function)
   - 4.2 [Computing the Style Cost](#42-computing-the-style-cost)
     - 4.2.1 [Style Matrix](#421-style-matrix)
     - 4.2.2 [Style Cost](#422-style-cost)
   - 4.3 [Defining the Total Cost to Optimize](#43-defining-the-total-cost-to-optimize)
5. [Solving the Optimization Problem](#5-solving-the-optimization-problem)
   - 5.1 [Load the Content Image](#51-load-the-content-image)
   - 5.2 [Load the Style Image](#52-load-the-style-image)
   - 5.3 [Randomly Initialize the Image to be Generated](#53-randomly-initialize-the-image-to-be-generated)
   - 5.4 [Load Pre-trained VGG19 Model](#54-load-pre-trained-vgg19-model)
   - 5.5 [Compute Total Cost](#55-compute-total-cost)
     - 5.5.1 [Compute Content Cost](#551-compute-content-cost)
     - 5.5.2 [Compute Style Cost](#552-compute-style-cost)
   - 5.6 [Train the Model](#56-train-the-model)
6. [Test With Your Own Image](#6-test-with-your-own-image)
7. [Results](results)

## 1. Packages

First, we import the necessary packages and dependencies for Neural Style Transfer. These include libraries for image processing, numerical computations, and deep learning.

## 2. Problem Statement

Neural Style Transfer (NST) is an optimization technique in deep learning that combines two images: a content image (C) and a style image (S) to create a generated image (G) that merges the content of C with the style of S.

## 3. Transfer Learning

We use the VGG-19 model, pre-trained on the ImageNet database, to build the NST algorithm. This model is used to extract features from the content and style images. Transfer learning leverages the pre-trained model's ability to capture intricate features, saving time and computational resources.

## 4. Neural Style Transfer

### 4.1 Computing the Content Cost

#### 4.1.1 Make Generated Image G Match the Content of Image C

To match the content of the generated image G to the content image C, we choose a middle activation layer of the VGG network. This layer captures both low-level features (like edges) and high-level features (like shapes).

#### 4.1.2 Content Cost Function

The content cost function ensures that the content in the generated image G closely matches the content of image C. It does this by minimizing the difference between the feature representations of G and C in the chosen layer.

### 4.2 Computing the Style Cost

#### 4.2.1 Style Matrix

The style of an image is represented by the Gram matrix, which captures the correlations between different filter responses. This matrix helps in understanding the texture and patterns of the style image.

#### 4.2.2 Style Cost

The style cost function minimizes the difference between the Gram matrix of the style image S and the Gram matrix of the generated image G. This ensures that G captures the stylistic patterns of S.

### 4.3 Defining the Total Cost to Optimize

The total cost function combines the content and style costs. This is done by weighting them with factors alpha and beta, respectively. The combined cost guides the optimization process to generate an image that balances both content and style.

## 5. Solving the Optimization Problem

### 5.1 Load the Content Image

We start by loading the content image, which is the base image whose content we want to preserve in the generated image.

### 5.2 Load the Style Image

Next, we load the style image, which provides the artistic style that we want to apply to the content image.

### 5.3 Randomly Initialize the Image to be Generated

The generated image G is initialized randomly. During the optimization process, this image will be adjusted to minimize the total cost function.

### 5.4 Load Pre-trained VGG19 Model

The pre-trained VGG-19 model is loaded. This model will be used to compute the feature representations needed for the content and style costs.

### 5.5 Compute Total Cost

#### 5.5.1 Compute Content Cost

The content cost is computed based on the difference in feature representations between the content image C and the generated image G in the chosen layer.

#### 5.5.2 Compute Style Cost

The style cost is computed by comparing the Gram matrices of the style image S and the generated image G across several layers.

### 5.6 Train the Model

The optimization process involves iteratively adjusting the pixels of the generated image G to minimize the total cost function. This is done using a gradient descent algorithm, where the gradients of the cost function with respect to the image pixels are used to update G.

## 6. Test With Your Own Image

You can test the NST algorithm with your own content and style images by replacing the default images in the code. This allows for customization and exploration of different artistic effects.

# Usage
To use the neural style transfer implementation, follow these steps:

1. Place your content image & style image in the `images/` directory.
2. Update the paths to these images in the notebook.
3. Run the notebook cells sequentially to generate the styled image.
4. The generated image will be saved in the `output/` directory.

# Results
Here are some examples of generated images using different content and style combinations:
- The beautiful ruins of the ancient city of Persepolis (Iran) with the style of Van Gogh (The Starry Night)
   ![Image](/images/perspolis_vangogh.png)

- The tomb of Cyrus the great in Pasargadae with the style of a Ceramic Kashi from Ispahan.
    ![Image](/images/pasargad_kashi.png)

- A scientific study of a turbulent fluid with the style of a abstract blue fluid painting.
    ![Image](/images/circle_abstract.png)
---
