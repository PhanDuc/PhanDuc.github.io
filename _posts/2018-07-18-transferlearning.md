---
title: "Deep Learning Project: Transfer learning"
date: 2018-07-18
tages: [machine lerning, data science, deep learning, neural network]
header:
    image: "/images/perceptron/percept.jpg"
excerpt: "Deep Learning, Machine Learning, Data Science"
mathjax: "true"
toc: true
toc_label: "Unique Title"
toc_icon: "heart"  
---

# Introduction
In this post, I want to write about how I can perform transfer learning using the official pre-trained model offered by Google, which can be found in TensorFlow’s model library [link](https://github.com/tensorflow/models/tree/master/research/slim).
This project is a part of my summer industrial internships at Inspector Cloud Company[link](https://inspector-cloud.ru/), where they are a successful company in retail. The goal of this project was to build a Deep Learning model that can recognize genuine and spoof image.

## Building DataSet 
They had many genuine images but very few spoofing image, the first thing I need to do before use any deep learning model that I needed to increase the number of spoof image in the different ways. It is essential to develop robust and efficient spoof detection that generalize well to new imaging conditions and environments. 
The progress to build spoof dataset can divided into 2 steps:
1.	Considering variability and generate more images using different devices and screens 
2.	Used data augmentation to helped us generated more images and from that changing the appearance of images slightly then making 10 random crops before passing them into the network for training.
Below is the number of device I used in this project
<img src="{{ site.url }}{{ site.baseurl }}/images/devices_table.jpg" alt="">

**Note that: Each camera will take approximately 200-300 images from different screens in different light condition**

Example images of spoof:
Used Iphone SE to took photos from Samsung A7 in outdoor light condition
<img src="{{ site.url }}{{ site.baseurl }}/images/fake1.jpg" alt="linearly separable data">
Oppo F5 camera and Ipad Air 1 screen, indoor light
<img src="{{ site.url }}{{ site.baseurl }}/images/fake2.jpg" alt="linearly separable data">
Samsung A7 camera and Monitor HP, under sun light.
<img src="{{ site.url }}{{ site.baseurl }}/images/fake3.jpg" alt="linearly separable data">
Image from the book product, taken by Samsung A7 (2016)
<img src="{{ site.url }}{{ site.baseurl }}/images/fake4.jpg" alt="linearly separable data">

To understand data augmentation applied to computer task, I decided to visualize a given input being augmented and distorted. In this step, I made Python script that uses Keras to perform data augmentation.
'''python
aug = ImageDataGenerator(rotation_range     = 30,
                             width_shift_range  = 0.2,
                             height_shift_range = 0.2,
                             shear_range = 0.2,
                             zoom_range  = 0.2,
                             horizontal_flip = True,
                             channel_shift_range= 0.15,
                             fill_mode       = "nearest")

total = 0
#print("[INFO] generating images...")
imageGen = aug.flow(image,
                    batch_size=1,
                    save_to_dir = args["output"],
                    save_prefix = os.path.splitext(file)[0],
                    save_format="jpg")

for image in imageGen:
    total += 1
    # Break when reached  10 examples
    if total == 10:
        break
'''
### H3 Heading

Content of the post i wrote

*text*

**bold**

What about a [link](https://github.com/phanduc)


Code block:
'''python
import numpy as np
'''

inline code 'x + y + z'

