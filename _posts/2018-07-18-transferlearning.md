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

