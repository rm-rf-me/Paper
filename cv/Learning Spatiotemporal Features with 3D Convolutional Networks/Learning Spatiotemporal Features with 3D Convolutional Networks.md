# Learning Spatiotemporal Features with 3D Convolutional Networks

### ICCV 2015 C3D

## Abstract

* 作者提出一个简单但是有效的使用3D卷积在大型监督视频数据集上训练的时空特征抽取模型。
* 作者有三点发现：
  * 3D卷积在时空特征抽取上比2D更加稳定。
  * 小如$3\times3\times3$的卷积核效果最好。
  * 该模型命名为C3D，达到了当时的SOTA。
* 并且这个模型非常小巧，仅用了10 dimensions就在UCF上达到了52.8%，可谓是短小精悍。

## 1.Introduction

* 一个好的video descriptor应该具有以下四点特征：
  * 更普适generic，需要适用于各种类型的视频。
  * 更简洁compact。
  * 高效计算dfficient。
  * 简单simple。
* 3D卷积能够很好的概括视频中的各种信息，包括物体、场景、动作等内容，并且能够直接应用到各种下游任务中，不需要额外的finetune。并且满足上述的所有特征。
* 本文的贡献包括：
  * 实验证明3D卷积在对外形和动作的特征去抽上表现良好。
  * 小卷积核效果最好。
  * 该模型在4个任务6个benchmark中效果最好。

## 2.Related Work

* 之前有很多传统的方式，比如：STIPs、SIFT-3D、HOG3D、Cuboids、ActionBank、Dense Trajectories（iDT），其中iDT是当前的SOTA。

## 3.使用3D卷积抽取特征

### 3.1 3D卷积和pooling

* 作者坚信[这篇](Large-scale video classiﬁcation with convolutional neural networks.)达到sota的原因是使用了3d卷积，从而没有丢失时序信息。
* 在2d的经验来看，小核深网络的效果会更好。这一点在3d可能同样适用。

#### Common network setting：

* 在UCF视频时，所有的视频帧都被缩放到128*171，这几乎是原分辨率的一半。
* 视频被切分到不重叠的16帧clips作为模型输入，所以输入的尺寸为$3\times16\times128\times171$。同时还会做一些jittering，具体为随机选取大小为$3\times16\times112\times112$的crops。
* 整个网络有5个卷积层和5个pooling层，并且pooling是紧接在conv之后的。两个全连接层和译者softmax loss层。卷积层的filter数分别为：64、128、256、256、256。
* 所有的卷积核都有参数d，代表了核的时序深度。
* 所有卷积层都使用了合适的padding，步长均为1。
* 所有的pooling都是max pooling，pooling核都为$2\times2\times2$（除了第一层），步长为1，这代表着输出随着因子8递减。第一个pooling层使用$1\times2\times2$，这主要是避免太早的融合时序信息。
* 两个全连接层输出都是2048。
* 训练使用mini-batch of 30 clips，学习率为0.003，并且学习率在4个epochs之后缩小到十分之一。最终的训练在16个epochs之后停止。

#### Varying network architectures：

* 主要出于研究如何更好的整合时序信息，作者只vary了时序核的深度$d_i$。作者实验了两种不同的结构：
  * homogeneous temporal depth：所有conv层的核尺寸都相同。
  * varying temporal depth：核尺寸在变化。
* 对于homogeneous来说，作者尝试了1、3、5、7，其中的1就等价于2d卷积。
* 对于varying来说，作者尝试了提升版：33557和下降版：75533。每一种结构都具有相同的输出和全连接层参数数量。参数量差距仅仅在卷积层所体现，但这比起全连接中百万级别的参数量来讲就微不足道了。

### 3.2 Exploring kernel temporal depth

* 作者在UCF上做了训练。
* 分析结果可知：
  * 核不变的实验中：
    * 3尺寸效果最好，收敛速度最快。
    * 1尺寸效果最差，因为其本质为2d卷积，缺少了时序特征。
  * 核变化实验中：
    * 3尺寸同样最好，但是三种方法的差距就没有那么大。
  * 作者还尝试了高分辨率，但结果基本相似。
* 总的来说，三核天下第一。

### 3.3 Spatiotemporal feature learning

#### Network architecture：

* 既然三核天下第一，那么只要算力和内存允许，就能够用三核堆到天荒地老。

* 作者的GPU环境下，用了八个卷积层，5个pooling层，两个全连接层核一个softmax层。具体的模型如下，作者给它起了个响亮的名字：C3D。

  | 层      | 尺寸 |
  | ------- | ---- |
  | conv1a  | 64   |
  | pool1   |      |
  | conv2a  | 128  |
  | pool2   |      |
  | conv3a  | 256  |
  | conv3b  | 256  |
  | pool3   |      |
  | conv4a  | 512  |
  | conv4b  | 512  |
  | pool4   |      |
  | conv5a  | 512  |
  | conv5b  | 512  |
  | pool5   |      |
  | fc6     | 4096 |
  | fc7     | 4096 |
  | softmax |      |

#### Dataset

* C3D模型在Sports-1M上训练过，

#### Training

* 训练在Sports-1M上训练完成。该数据集有很多长视频，所以作者在每个训练数据中随机抽取五个两秒的clips。clips被缩放到每帧128*171。
* 在训练过程中，随机crop输入的clip到$16\times112\times112$。同时还做了50%几率的horizontally flip。
* 训练使用SGD，mini-batch为30。学习率为0.003，并且在150K iterations之后减少二分之一。训练大约停止在1.9M iterations（大约13epochs）。
* C3D网络除了从零开始训练，作者还尝试了finetune在I380K上pretrain的版本。

#### Sports-1M classification results

* C3D使用非常简单的采样方式和小的代价达到了一个比较好的效果。

#### C3D video descriptor

* 经过训练之后，该网络能够当作特征抽取器应用到别的任务中。

### what does C3D learn

* 作者使用了deconvolution的方法是图去理解模型学到了什么。观察到模型先在开始的几帧内关注外形的特征，然后在后续帧内关注动作特征。

## 4. Action recognition

### Dataset

* UCF101。

### Classification model

* 本文抽取了C3D的特征，将其输入给了SVM中。

* 作者尝试了三种训练方式：

  * trained on I380
  * trained on Sports-1M
  * trained on I380 and finetined on Sports-1M

  其中都使用了L2正则化。

### Baselines

* 比对了几种baselines：
  * 当前最好的手工features iDT
  * 最常用的deep image features Imagenet

### Results

* C3D只需要4096维就能够达到82.3%。如果增加到12288维就能达到85.2%。而融合了iDT之后就能够达到90.4%。

### C3D is compact

* 为了验证C3D特征的紧凑性，作者使用了PCA将特征压到低维然后用SVM测试acc。同样的还对iDT和Imagenet做了同样的操作。
* 在很极端的10维时C3D达到了52%，比剩下两个高了二十和三十个点。在50和100维时达到了72和75，比剩下两个高了10个点左右。最后在500维时达到了79.4%高出了5到是个点。
* 作者还降维分析了C3D最后特征的分布，能够较为明显的看出不同类的聚集现象。这也让不经过finetune直接用成为可能。

## Action Similarity Labeling

### Dataset

* ASLAN。
* 该任务是需要判别给定的两个视频是否属于同一类别。该类别只关心是否相似而不关心具体的动作到底是什么。这个任务很具有挑战性，因为任务中包含一个never seen before类。

### Features

* 作者split视频到16帧clips并且有8帧的overlap。
* 对每个clips提取了prov、fc7、fc6、pool5的特征。The features for videos are computed by averaging the clip features separately for each type of feature, followed by an L2 normalization.

### Classification model

* 给定一对视频，计算12种不同的距离，这都与[这篇](The action similarity labeling challenge.）相同。
* 对于四种feature，每种12维，所以每对视频都得到了48的特征向量。这些向量不是直接比较的，二十分别做了normalize。最后用了SVM去区分这些48维的向量是否相同。

### Results

* 当前方法大多都使用多种手工标注的feature和strong encoding方法，和复杂的学习模型。而本文就是个简单的C3D和SVM。但达到了sota并且高了十个点左右。

## Scene and Object Recognition

### Datasets

* YUPENN：420 videos，14 scene categories。
* Maryland：130 videos，13 scene categories。

### Classification model

* 对于两个数据集，作者使用了相同的特征提取器和SVM。
* 对于object数据集来说，标准的evaluation是基于帧的。不过C3D使16帧的clip来抽取特征。作者选用每个clip里最频繁出现的label作为ground truth。如果最频繁的标签出现次数小于8帧，则认为这是一个negative的clip，并且没有object，所以直接删掉。

### Results

* Maryland上提高了10个点，YUPENN更是直接刷到98了。
* C3D就用了个小破SVM，还是半分辨率输入，还么得finetune。这简直就是吊着打。