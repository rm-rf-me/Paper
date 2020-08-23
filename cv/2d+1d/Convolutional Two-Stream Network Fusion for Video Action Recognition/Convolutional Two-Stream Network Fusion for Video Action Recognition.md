



# Convolutional Two-Stream Network Fusion for Video Action Recognition

### 2016 CVPR

## Abstract

* 作者研究了诸多的融合方式，有了以下发现：
  * 相比于在softmax层做融合，一个时空网络在卷积层融合能够达到相同的效果，但是却可以节省许多参数开销。
  * 在靠后的层做融合效果要更好。同时在分类层添加额外的融合能够提升整体效果。
  * 在时间和空间信息交汇的位置对抽象的卷积特征做pooling能够显著提升模型表现。
* 基于这些作者设计了新的CNN结构来完成视频的时空信息融合，并达到SOTA。

## 1.Introduction

* CNN在计算机视觉领域成果颇多，但在video recognition方面表现并不好。作者认为原因首先可能是数据集太小或者噪声太多。
* 相比于图像分类，视频中的动作分类具有更高的不确定性，所以理应有更大的数据集，ImageNet中对每个类别有1000个以上的样本，而UCF101每个类只有少的可怜的100个。
* 另外一个重要原因是CNN不能够很好的捕捉到时序信息，往往都退化成为了图像分类。
* 之前往往是使用双流法来完成任务，但是双流结构不能很好的抓住两点重要的线索：
  * 谁在向哪里运动，即时间线索和空间线索。就是说模型对于两类信息只是单独的思考和简单融合，并不能够充分结合空间和时间总和考虑问题。
  * 双流法对信息的利用很少，空间只使用一帧，时间只使用十帧。
* 本文主要研究了：如何进行空间融合，如何进行时间融合。

## 2.Related work

* 一个很直观的解决时序信息问题的方法就是在时间维度上堆叠多个2d视频帧。有文章研究了多种时序采样方法，包括early fusion，slow fusion和late fusion。但这些结构的效果和单纯的空间信息模型很相似，意味着模型并没有学到太多时序信息。

* 最近的C3D方法应用$3*3*3$的3d卷积到16个连续帧中，得到了更好的效果。但是从模型结构来看，最大的不同只是变得更深而没有本质改变。

* 另一种时序关系方法是拆分3d卷积到2d空间和1d时间卷积，如[这篇](Human Action Recognition using Factorized Spatio-Temporal Convolutional Networks)。

* [这篇](Beyond Short Snippets: Deep Networks for Video Classification)讨论了多种用来结合长时间时序特征的pooling结构，文章认为卷积层的时序pooling要好于slow、local、late等pooling。文章还调查了使用LSTM整理过的序列并没有给CNN features的temporal pooling带来多大提升。

* 和本文最相关的就是经典的双流法paper，该方法首次解耦了时间和空间信息，这些信息被分别传入CNN结构，学习事物的空间和时间信息。每个流都单独的进行video recognition，最后通过late fusion softmax分数来得到分类。作者对比了多种光流方法，最终得出simple stacking of L = 10 horizontal and vertical flow fields即水品和垂直l都为10的效果是最好的。作者还在UCF101和HMDB51上做了multitask learning来扩充数据集。截止到现在，双流法是action recognition效果最好的方法，尤其是在小数据集上。

* 和本文相关的还有[bilinear method](Bilinear CNNs for Fine-grained Visual Recognition)，分别使用两个CNN层抽取特征，然后对图像的每个位置坐outer product融合，最后the resulting bilinear feature is pooled across all locations init an orderless descriptor。本文中主要涉及的关联是second-order pooling of hand-crafted SIFT feature。

* 对于数据：

  * Sports-1M有非常多的数据（大约1M）和类别（487），所以不可避免的就是噪声。
  * THUMOS有大于45M帧，但是只有一小部分是有标签的。

  因为label noise，对于时空信息CNN的训练还要依靠UCF101和HMDB51这样的短视频，但同样要面临过拟合的风险。

## 3.Approach

* 原始双流法有两个主要缺点：

  * 不能在像素的角度学到空间流和时间流的关联关系。
  * 空间仅为单帧，时间仅为一小堆帧（如10帧）。

  原文中对于第二个问题采用了时序pooling的方法，但这样仍有局限。

### 3.1.Spatial fusion

* fusion要做的就是融合两个网络的信息，使得在判别每个像素值的时候能够总和考虑两个网络的结果。比如判别刷牙和梳头动作时，时序网络判别出同样的刷/梳这个动作，然后空间网络定位这个动作的位置是在牙还是头发上。

* 简单的融合方法就是用一个网络的层overlaying/stacking另一个网络的层，但当一个网络中的一些channel和另一个网络的一些channel做corresponds的时候还会存在问题。

* 我们假设在空间流中不同的channel代表了不同的面部部位，在时间流中不同channel代表了不同模式的运动。所以，在channel被stacked之后，后续的层必须能够学习到被融合的两种不同的信息。

* 更具体的，作者将详细讨论不同的融合方法。首先规定：

  * 融合函数：$f: \mathbf{x}_{t}^{a}, \mathbf{x}_{t}^{b}, \rightarrow \mathbf{y}_{t}$
  * 空间输入：$\mathbf{x}_{t}^{a} \in \mathbb{R}^{H \times W \times D}$
  * 时间输入：$\mathbf{x}_t^b \in \mathbb{R}^{H \times W \times D}$

* | 融合方式             | 表达式                                                       | 维度                                                         | 增加参数                            |
  | -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------- |
  | Sum fusion           | $\mathbf{y}^{sum}=\mathbf{x}^a_t + \mathbf{x}_t^b$           | $\mathbf{y}^{sum}\in \mathbb{R}^{H\times W\times D}$         | 无                                  |
  | Max fusion           | $\mathbf{y}^{max}=max(\mathbf{x}_t^a,\mathbf{x}_t^b)$        | $\mathbf{y}^{max}\in \mathbb{R}^{H\times W\times D}$         | 无                                  |
  | Concatenation fusion | $\mathbf{y}^{cat}=cat(\mathbf{x}_t^a,\mathbf{x}_t^b)$        | $\mathbf{y}^{cat}\in \mathbb{R}^{H\times W\times 2D}$        | 全连接层                            |
  | Conv fusion          | $\mathbf{y}^{conv}=\mathbf{y}^{cat} * f + b$                 | $f\in \mathbb{R}^{1\times 1\times 2D \times D}$$\mathbf{y}^{conv}\in \mathbb{R}^{H\times W\times 2D}$ | 卷积核参数和偏置                    |
  | Bilinear fusion      | $\mathbf{y}^{\mathrm{bil}}=\sum_{i=1}^{H} \sum_{j=1}^{W} \mathbf{x}_{i, j}^{a \top} \otimes \mathbf{x}_{i, j}^{b}$ | $\mathbf{y}^{bil}\in \mathbb{R}^{D\times D}$                 | 用SVM代替了全连接层，反而减少了参数 |
  
  Bilinear fusion的主要问题在于特征是高维度的，所以经常会加入ReLU5，全连接层会被去掉，并且使用power-和L2-normalisation来增强SVM的分类能力。该方法的优点在于每个channel都和全部的channel结合过，而缺点就是空间信息变得不是那么直观。
  
* 最终的结果为：

  | 方法          | 位置    | Acc    | 层数 |   参数    |
  | ------------- | ------- | ------ | ---- | :-------: |
  | Sum（原双流） | Softmax | 85.6%  | 16   |  181.42M  |
  | Sum（ours）   | Softmax | 85.94% | 16   |  181.42M  |
  | Max           | ReLU5   | 82.70% | 13   |  97.31M   |
  | Concatenation | ReLU5   | 83.53% | 13   |  172.81M  |
  | Bilinear      | ReLU5   | 85.05% | 10   | 6.61M+SVM |
  | Sum           | ReLU5   | 85.20% | 13   |  97.31M   |
  | Conv          | ReLU5   | 85.96% | 14   |  97.58M   |

* 注意到：

  * max、sum和conv参数量几乎一致，conv因为额外的卷积核略多。
  * concatenation在融合后没有降维，参数量几乎为两倍。

### 3.2 融合位置

* 理论上可以在任何位置融合。并且假设两个网络的尺寸是相同的，这点可以通过upconvolutional层或者小维度直接pad 0来实现。

* 融合位置比较：

  | 位置            | Acc    | layers | 参数    |
  | --------------- | ------ | ------ | ------- |
  | ReLU2           | 82.25% | 11     | 91.90M  |
  | ReLU3           | 83.43% | 12     | 93.08M  |
  | ReLU4           | 82.55% | 13     | 95.48M  |
  | ReLU5           | 85.96% | 14     | 97.57M  |
  | ReLU5+FC8       | 86.04% | 17     | 181.68M |
  | ReLU3+ReLU5+FC6 | 81.55% | 17     | 190.06M |

* 发现：

  * 早融合不太行。
  * 早融合+多融合也不太行，甚至还不如单融合。
  * 晚融合和多融合效果最好，但是参数多一倍。

### 3.3时间融合

* 三种方法：

  * 2D conv，2D pooling：在时间上平均预测值。

  * 2D conv，3D pooling：时间维度上的帧堆叠之后进行pooling，注意这里的堆叠是指channels的堆叠，不会跨channels做pooling。

  * 3D conv，3D pooling：先使用$f \in \mathbf{R}^{W^{''} \times H^{''} \times T^{''} \times D \times D^{''}}$的卷积和${b}\in \mathbf{R}^D$的偏置做3D卷积：
    $$
    \mathbf{y} = \mathbf{x} * \mathbf f + b
    $$

* 效果为：

  | fusion | pooling | 位置   | UCF101 | HMDB51 |
  | ------ | ------- | ------ | ------ | ------ |
  | 2D     | 2D      | ReLU5+ | 89.35% | 56.93% |
  | 2D     | 3D      | ReLU5+ | 89.64% | 57.58% |
  | 3D     | 3D      | ReLU5+ | 90.40% | 58.63% |

### 3.4 Proposed architecture

* 具体模型示意图见[这里](https://blog.csdn.net/u013588351/article/details/102074562)。
* 在最后一个卷积层即ReLU5处融合，将时间流用3D conv加进空间流中并3Dpooling。同时并没有丢掉时间流，而是同样使用3D pooling。
* 在训练时两个losses都会被用到。在测试时会对两个流的结果取均值。
* 抽帧步长为$\tau$，光流覆盖帧数为L，抽帧数量为T。如果$\tau < T$则会出现重叠，否则不会重叠。

### 3.5 Implementation details

#### 双流结构：

* 作者使用了两个在ImageNet上pre-trained的模型。时序网络同样在ImageNet上pre-train过。
* 输入帧都经过缩放处理，最短边长度为256。
* 没有使用batch normalization。

#### 双流卷积fusion：

* 网络finetune时使用96的batch size，学习率从$10^{-3}$开始，随着验证集acc饱和逐渐下降。作者只反向传播到新加入的fusion层，因为整体的发香传播对结果没有什么帮助。

#### Spatiotemporal architecture：

* 最终模型的3Dfusion核f维度为：$3\times 3\times 3\times 1024\times 512$并且T=5，也就是说卷积核为$3\times 3\times 3$，1024是将时间和空间流相融合的结果，512为之后FC6的输入尺寸。
* 3D卷积相比于2D卷积更容易过拟合。在finetune中，每次iteration中对batch中96个视频都随机抽取T=5帧，然后在1～10之间选择$\tau$，所以总的抽样帧数在15到50之间。
* 作者随机的25%jitter输入的长和宽，然后rescale会224*224。crop的位置（size，scale，horizontal flipping）在第一帧中随机选择，然后同步到之后的所有帧中。

## 4.Evaluation

### 4.1 数据和experimental protocols

* UCF101和HMDB51.
* provided evaluation protocol并且report the mean average acc over three splits into training和test data

### 4.2 空间上融合两个流

* 在实验中作者使用了相同的CNN结构，比如VGG-M-2048。fusion层被放在了最后一个卷积层之后，也就是说该层的输入是两个流的ReLU5的输出。

### 4.3 融合位置

* 在ReLU5的效果已经很好了，虽然再加上FC8能够更好一点点，但参数量翻了一倍，这就很不划算。

### 4.4 Going from deep to very deep models

* 出于算力原因，之前的网络都使用类似VGG-M-2048这样的CNN。实际上更深的网络能有更好的效果，所以作者使用了VGG-16。模型都在ImageNet上做了pre-train，只有HMDB51的时间网络是使用UCF的时间网络初始化的。作者使用了相同的3D卷积训练。但是额外的增加了图像中心的抽样。学习路从50^-4岁者验证集收敛下降。
* 最后的比对中发现，更深的网络在空间上大概能有10个点的提升，在时间上能有4个点的提升，总体能有四五个点的提升。

### 4.5 如何在时间流融合

* 具体的方法和效果已经在3.3中分析过了。

### 4.6 与SOTA比较

* 用八个点吊打LRCN（这篇我也写过），五个点揍了C3D，两个点锤了双流的各种改进版本。总的吧UCF拉到92%，HMDB拉到65%。
* 并且本文的fusion方法融合FB encoded IDT特征达到了更好的分数。