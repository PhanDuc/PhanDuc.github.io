---
title: "Deep Learning Project: Image Anti-Spoofing"
date: 2018-07-18
tags: [deep learning]
header:
    image: "/images/perceptron/percept.jpg"
excerpt: "Deep Learning, Machine Learning, Data Science"
toc: true
toc_label: "Table of Content"
toc_icon: "heart"  
mathjax: "true"
---

## 1. Introduction
In this post, I want to write about how I can perform transfer learning using the official pre-trained model offered by Google, which can be found in TensorFlow’s [model library](https://github.com/tensorflow/models/tree/master/research/slim).
This project is a part of my summer industrial internships at [Inspector Cloud](https://inspector-cloud.ru/), where they are a successful company in retail. The goal of this project was to build a Deep Learning model that can recognize genuine and spoof image.

## 2. Building DataSet 
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

This is 10 fake images that generated from one image
<img src="{{ site.url }}{{ site.baseurl }}/images/10_img_aug.jpg" alt="linearly separable data">



Then from each output images, I made 10 crop in random location with the size of each small image is 
* Width = Width of Original Image // 4
* Height = Width of Original Image // 4

There are two benefit with this crop,
1.	Increase the number of images
2.	Pretrained models given input images in square

I repeated that steps with all the pairs of camera and screen I had. Then after all I can generated around **200,000** spoof images.

Now we are good to move to the next step!

## 3. Build The Model

I decided to used some pre-trained models then compare between to choice the best one.

Here is the pre-trained model that I used to tested:

* inception_v3 [paper](http://arxiv.org/abs/1512.00567) - [download](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)
* inception_v4 [paper](http://arxiv.org/abs/1602.07261) - [download](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz)
* inception_resnet_v2 [paper](http://arxiv.org/abs/1602.07261) - [download](http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz)

This picture below showed 2 differents type of transfer learning 
(thank [ronny](http://ronny.rest/blog/post_2017_10_13_tf_transfer_learning/) for great picture)
<img src="{{ site.url }}{{ site.baseurl }}/images/type_of_transfer_learning.jpg" alt="linearly separable data">

In this project I used method `A2`, which is, use all the layers from the pre-trained model, except the final layer, creating a new output layer that is similar in structure to the original, but adapted to train on a new classification task. The layers from the pre-trained model will remain trainable, and continue to be fine-tuned for the new task.

Because I only had 2 classes, they are: real image and fake image.
Then I used the final layer sigmoid activation, which is perfect for a binary classification.
<img src="{{ site.url }}{{ site.baseurl }}/images/sigmoid.jpg" alt="linearly separable data">

Traning process....

## 4. Result

....I had several models after training completed, now I will do **Test Time Augmentation(TTA)**. What this means is that I'm going to take `x` crop at random, then calculate predictions for all these images, take the average, and make that final prediction.

Each model will performs 3 differents type of **TTA**:
1. Take 5 crops at random
2. Take 10 crops at random
3. Take one center-cropped

This is the result of all the models
**Please note that, this result done in test dataset that all the models never seen before!**

```
======== inception_resnet_v2_resize_20000_tta10.pb ========
	Precision: 0.997
	Recall: 0.620
	F1: 0.765

             precision    recall  f1-score   support

          0       0.07      0.94      0.14       623
          1       1.00      0.62      0.76     19371

avg / total       0.97      0.63      0.75     19994

======== inception_v4_resize_12902_tta5.pb ========
	Precision: 1.000
	Recall: 0.011
	F1: 0.022

             precision    recall  f1-score   support

          0       0.03      1.00      0.06       623
          1       1.00      0.01      0.02     19371

avg / total       0.97      0.04      0.02     19994

======== inception_v4_resize_12902_notta.pb ========
	Precision: 1.000
	Recall: 0.018
	F1: 0.035

             precision    recall  f1-score   support

          0       0.03      1.00      0.06       623
          1       1.00      0.02      0.03     19371

avg / total       0.97      0.05      0.04     19994

======== inception_resnet_v2_resize_12984_notta.pb ========
	Precision: 0.997
	Recall: 0.643
	F1: 0.781

             precision    recall  f1-score   support

          0       0.08      0.93      0.14       623
          1       1.00      0.64      0.78     19371

avg / total       0.97      0.65      0.76     19994

======== inception_v3_resize_12122_tta5.pb ========
	Precision: 0.981
	Recall: 0.965
	F1: 0.973

             precision    recall  f1-score   support

          0       0.27      0.41      0.33       623
          1       0.98      0.97      0.97     19371

avg / total       0.96      0.95      0.95     19994

======== inception_v3_resize_12122_tta100.pb ========
	Precision: 0.996
	Recall: 0.455
	F1: 0.625

             precision    recall  f1-score   support

          0       0.05      0.95      0.10       623
          1       1.00      0.46      0.63     19371

avg / total       0.97      0.47      0.61     19994

======== inception_resnet_v2_resize_20000_notta.pb ========
	Precision: 0.998
	Recall: 0.524
	F1: 0.687

             precision    recall  f1-score   support

          0       0.06      0.97      0.12       623
          1       1.00      0.52      0.69     19371

avg / total       0.97      0.54      0.67     19994

======== inception_resnet_v2_resize_20000_tta100.pb ========
	Precision: 0.998
	Recall: 0.663
	F1: 0.796

             precision    recall  f1-score   support

          0       0.08      0.96      0.15       623
          1       1.00      0.66      0.80     19371

avg / total       0.97      0.67      0.78     19994

======== inception_resnet_v2_resize_20000_tta5.pb ========
	Precision: 0.982
	Recall: 0.951
	F1: 0.966

             precision    recall  f1-score   support

          0       0.23      0.45      0.30       623
          1       0.98      0.95      0.97     19371

avg / total       0.96      0.94      0.95     19994

======== inception_v3_resize_12122_tta10.pb ========
	Precision: 0.996
	Recall: 0.416
	F1: 0.587

             precision    recall  f1-score   support

          0       0.05      0.95      0.09       623
          1       1.00      0.42      0.59     19371

avg / total       0.97      0.43      0.57     19994

======== inception_resnet_v2_resize_12984_tta10.pb ========
	Precision: 0.996
	Recall: 0.670
	F1: 0.801

             precision    recall  f1-score   support

          0       0.08      0.92      0.15       623
          1       1.00      0.67      0.80     19371

avg / total       0.97      0.68      0.78     19994

======== inception_v4_resize_12902_tta10.pb ========
	Precision: 1.000
	Recall: 0.003
	F1: 0.007

             precision    recall  f1-score   support

          0       0.03      1.00      0.06       623
          1       1.00      0.00      0.01     19371

avg / total       0.97      0.03      0.01     19994

======== inception_v3_resize_12122_notta.pb ========
	Precision: 0.992
	Recall: 0.249
	F1: 0.399

             precision    recall  f1-score   support

          0       0.04      0.94      0.07       623
          1       0.99      0.25      0.40     19371

avg / total       0.96      0.27      0.39      19994
```

After that, the leader decided to wait until we have more unseen of fake data from customer, then they will perform training again to make sure they got better result before making a decide which model should deploy to product.



## 5. Conclution

For me, this is very interesting project that I joined from the beginning. From prepare data to trained models then tested and made the final decision.
- Make sure we got the dataset that contains all the subjects we need and in different perspective that can enable to find correct solutions and increase the recall score.
- It is good to used Data Augmentation when your data is not big enough for pre-trained model.
- The pre-trained CNN models learns this task very quickly but you need to find the exactly pre-trained model by try several and compare the results from them.

Happy Coding! :)



