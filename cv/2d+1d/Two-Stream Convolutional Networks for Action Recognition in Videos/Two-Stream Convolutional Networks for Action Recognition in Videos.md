# Two-Stream Convolutional Networks for Action Recognition in Videos

### 2014年的上古版本

## Abstract

三大贡献：

* 提出two-stream ConvNet，是一种合并时空信息的网络。
* ConvNet在多帧小数据上良好训练的范例。
* 证明了multi-task learning既能增加数据又能提升训练结果。

## Introduction

* 传统动作检测方法都仅使用图片，实际上视频中能表达的信息远不止图片。

* 本项目旨在使用在图像表达中最为重要的CNN模型，虽然CNN已经在2014年不久前被应用在这里，但是效果明显差于手工抽取的表达方法。

* 作者提出了双流法：空间流和时间流相结合的方法。空间流依旧是视频帧。而时间流为密集光流dense optical flow。

* 通过解耦时间空间流网络为使用在ImageNet上pre-train的模型提供可能。

* 作者的模型基于双流假说two-stream hypothesis：

  > 人类的视觉皮层包括两部分：负责物体检测的ventral stream和负责辨认运动的dorsal stream。

## Related work

* video recognition深受image recognition方法的影响。整个video action recognition方法家族都基于对局部时序信息的浅层高维表示shallow high-dimensional。
  * 具体的，浅层高维表示算法包括对使用局部时序特征表示的离散时空关键点的检测。
  * 其中的局部时空特征包括：方向梯度矩阵Histogram of Oriented Gradients（HOG）和光流矩阵Histogram of Optical Flow（HOF）。
  * 接着时空特征被编码到Bag of Features（BoF）表示中：即先对全部的时空格子grids做pooling（类似于spatial pyramid pooling），然后使用SVM分类器合并。
* 为了避免在巨大的时空特征大方块上做计算，SOTA的shallow视频表征都充分利用了密集关键点的轨迹信息。最好的trajectory-based pipline是Motion Boundary Histogram（MBH）。近期在trajectory-based的提升包括对全局动作信息的补偿、Fisher向量编码及其变体。
* 业界当然企图尝试一些深度学习的方法，但都使用视频帧作为输入。其中的HMAX结构提出了在第一层使用pre-defined时空过滤器filters。不久之后该方法融合进了空间spatial HMAX模型中，从而融合了时空信息。而不同于本作的方法，HMAX模型仅使用人工标注且浅层的网络（3 layer）。
* 而在RBM、ISA等模型中使用了对时空特征的无监督学习，并在其中添加了动作判别器。并在一些大型的数据上比如Sports-1M训练。不过研究发现，该模型对单帧输入和多帧输入的表现并没有太大差距。也就是说模型根本没怎么学到动作信息。最终模型fine-tuned到UCF-101数据之后效果比hand-crafted的trajectory-based方法差了20%。

## Two-stream architecture for video recognition

* 视频信息能够分解为两种：时间和空间。空间即自然的物体形象。而时间信息涉及到镜头的运动和物体的运动。所以作者将模型结构同样分成两部分，每个都使用ConvNet、softmax分数并进行late fusion。同时作者尝试了两种fusion方法：averaging和使用L2正则化softmax分数作为特征训练的多分类线性SVM。

### Spatial stream ConvNet

* 空间流当然要抓图片信息。既然spatial ConvNet本质就是图片分类，所以直接用当时最先进的image recognition方法，并且在如ImageNet之类的大数据上作pre-train。反正就为所欲为呗。

### Optical flow ConvNets

##### ConvNet input configurations

* **Optical flow stacking**
  
  * 用来描述每个帧中固定点的光流变化。
  
  * 一个密集光流可以看作是连续帧之间堆叠的向量。
	
  * CNN的每个输入都截取长度为L的连续帧，并将其拆分为垂直和水平拼接为2L的输入。具体的：

    $$
    \begin{array}{l}I_{\tau}(u, v, 2 k-1)=d_{\tau+k-1}^{x}(u, v) \\ 		I_{\tau}(u, v, 2 k)=d_{\tau+k-1}^{y}(u, v), \quad u=[1 ; w], v=[1 ; h], k=[1 ; L]\end{array}
    $$
  
  * 对于任意点（u, v），I（u, v, c）中c取值在1～2L之间
  
* **Trajectory stacking**

  * 用来描述运动点在多个帧中的运动轨迹。

  * 一个可选的动作表达方式。具体的：
    $$
    \begin{array}{l}I_{\tau}(u, v, 2 k-1)=d_{\tau+k-1}^{x}\left(\mathbf{p}_{k}\right) \\ I_{\tau}(u, v, 2 k)=d_{\tau+k-1}^{y}\left(\mathbf{p}_{k}\right), \quad u=[1 ; w], v=[1 ; h], k=[1 ; L]\end{array}
    $$
    其中$p_k$为轨迹的第k个点。对于起点（u，v）：
    $$
    \mathbf{p}_{1}=(u, v) ; \quad \mathbf{p}_{k}=\mathbf{p}_{k-1}+\mathbf{d}_{\tau+k-2}\left(\mathbf{p}_{k-1}\right), k>1
    $$

* Bi-directional optical flow

  * 将连续的L帧分为两部分，当前帧之后的L/2做正向计算，当前帧之前的L/2做反向计算。

* Mean flow subtraction

  * 为了减小镜头移动所带来的影响。

## Multi-stak learning

* 空间流CNN能够在图像数据上做pre-train，但时间流必须在视频数据上训练。
* 而现行数据并不多，单一数据容易过拟合。
* 作者使用multi-task learning的方法来将两种数据结合起来。具体的是将两个数据的task结合。则非当前数据的tasks就充当正则化项。
* 整个模型最后有两个softmax分类层，分别对应两个数据集。并且每层都有各自的loss函数。

## Implementation details

* CNN：模型如下。隐藏层使用ReLU。max pooling在3*3步长为2.

  ![截屏2020-07-14下午4.24.41](/Users/liou/Desktop/截屏2020-07-14下午4.24.41.png)

* Training：mini-batch SGD，momentum=0.9。batch size=256。接256帧。。。

* Testing：25帧。

* Pre-train on ImageNet

* Multi-GPU training：caffe，4*Titan*1day。

* Optical flow：OpenCV。

## Evaluation

* UCF101和HMDB51。

#### Two-stream ConvNets

* 在融合时空流的时候。对两个softmax使用均值和SVM。SVM最好。