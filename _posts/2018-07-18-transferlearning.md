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

To understand data augmentation applied to computer task, I decided to visualize a given input being augmented and distorted. 
In this step, I made Python script that uses Keras to perform data augmentation.
```python
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
```
In here, I used some parameters:
* The `rotation_range` parameter control the degree range of the random rotation, allow input image to be randomly rotated 30 degree.
* Both `width_shift_range` and `height_shift_range` are used for horizontal and vertical shifts, respectively. The value is a fraction of the given dimension, 20% in this case
* The `shear_range` controls the angle in counterclockwise direction as radians in which image will allowed to be sheared.
* The `zoom_range`, make image to be zoomed in or zoomed out in the range of value: `[1 – zoom_range, 1 + zoom_range]`
* The `channel_shift_range`, will make input channel random shifts.
* The `horizontal_flip` boolean controls whether or not a given input is allowed to be flipped horizontally during the training process.
* And the last parameter, `fill_mode` will points outside the boundaries of the input are filled according to the given mode:

For more detail, please refer to [link](https://keras.io/preprocessing/image/)
Lets see how the resuls

This is a fake image



Then from each output images, I made 10 crop in random location with the size of each small image is 
* Width = Width of Original Image // 4
* Height = Width of Original Image // 4

There are two benefit with this crop,
1.	Increase the number of images
2.	Pretrained models given input images in square

I repeated that steps with all the pairs of camera and screen I had. Then after all I can generated around 200.000 spoof images.

Now we are good to move to the next step!

### Build The Model

I decided to used some pre-trained models then compare between to choice the best one.
Here is the pre-trained model that I used to tested:
* inception_v3[link](http://arxiv.org/abs/1512.00567) - download[link](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)
* inception_v4[link](http://arxiv.org/abs/1602.07261) - download[link](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz)
* inception_resnet_v2[link](http://arxiv.org/abs/1602.07261) - download[link](http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz)


*text*

**bold**

