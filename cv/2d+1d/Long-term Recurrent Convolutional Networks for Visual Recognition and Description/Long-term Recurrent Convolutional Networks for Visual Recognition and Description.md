# Long-term Recurrent Convolutional Networks for Visual Recognition and Description

### CVPR 2015

## Abstract

* CNN在图像表达任务中几乎一家独大。而本文就想要实验RNN类的模型能够在这些类任务中产生什么效果。
* 作者开发了一个不同寻常的recurrent convolutional结构。能够适用于大范围视觉理解，能够端到端训练。
* 同时作者示范了该模型在video recognition、image description、retrieval problem、video narration challenges等任务上的效果。
* 不同于当前的结合时空表征的或单纯在时序上取均值的方法。本项目方法为“doubly deep”，包括时间和空间层的双重深度。该特点在复杂任务和有限数据情况下优势明显。
* 当一些非线形部分被纳入神经网络中，模型就具备学到long-term关系的能力。使用Long-term RNN模型能够直接完成可变输入到可变输出到映射，如视频帧到文本。并且让模型动态维护时序信息。

## Introduction

* 对每个视频帧做CNN，然后整体过LSTM。端到端、变长输入输出。

## Background

* LSTM在序列数据上有两点优势：
  * 与视觉系统结合之后很容易做端到端fine-tune。
  * 并没有要求输入输出长度。

## Long-term Recurrent Convolutional Network（LRCN） model

* 文中的模型对一张图像或者一帧作为$v_t$，通过特征变形函数：$\phi_V(v_t)$来映射到定长向量表达$\phi_t$。从而组合成时序特征序列。之后经过序列模型得到序列输出。最后经过softmax得到输出。

* 这个非常深的object recognition模型的成功之处在于组合使用了很多的非线性函数，这让模型变得健壮。

* 作者考虑了三个任务：activity recognition、image description、video description。分别对应了三类任务：
  * 序列输入，定长输出：activity recognition就是很好的例子，输入为视频序列，而输出为动作标签。
  * 定长输入，序列输出：如image description。
  * 序列输入，序列输出：video description。
  
  在本文中：
  
  * 序列输入，定长输出：采用后融合late fusion来整合每个分段的label，得到最终的分类。
  * 定长输入，序列输出：直接将输入复制给每个时间步timesteps。
  * 序列输入，序列输出：encoder-decoder模型。经过该模型后可以认为整个系统有了$T+T^{'}$个时间步，先经过$T$步完成输入编码，随后经过$T{'}$步完成解码。
  
* 整个模型的参数$(V, W)$包括模型的视觉和序列参数，都能通过在基于$t$时刻获得的信息$(x_{1:t}, y_{1:t-1})$，来对$y_{t}$做极大似然估计完成优化。具体的，最小化negative log likelihood：
  $$
  \mathcal{L}(V, W)=-\sum_{t=1}^{T} \log P_{V, W}\left(y_{t} \mid x_{1: t}, y_{1: t-1}\right)
  $$
  使用带动量的SGD。

## Activity recognition

* 作者提出了LRCN的两种变体：
  * LSTM在CNN的第一个全连接层之后。（$LRCN-fc_6$）
  * LSTM在CNN的第二个全连接层之后。（$LRCN-fc_7$）
* LRCN网络选用16帧的clips作为输入，抽帧步长为8帧并做了平均。
* 同时作者还考虑了RGB输入和flow输入，flow输入被转化为flow image，包括x和y两个通道表达其在不同方向上的光流值，数值在128附近然后放缩到0～255之间。第三个通道计算流的数量级flow magnitude。
* CNN部分为AlecNet，模型在ImageNet的子集ILSVRC-2012上pre-train。

### Evaluation

* UCF-101。
* fc6即LSTM在第一个全连接层的模型效果更好。
* RGB和flow采用不同的融合比也会影响模型的效果，1:1并不是最优比例。

## Image description

* 静态图像的description仅需要一个单独的CNN。
* 整个模型采用了多个LSTM堆叠的结构。
* 在每一个时间步内，图像和之前的输出词都被拼接为输入。在时间步t时，底层的LSTM输入为embedded的对于上一步的ground truth word，对于序列生成而言，输入就是之前的模型输出。
* 对于LSTM栈中的第二个LSTM层来说，该层的输入融合了最底层的LSTM输出和图像的描述image representation，其中的representation使用了Caffe中预设的模型，非常类似AlexNet，并在ILSVRC上pretrain。
* 之后的LSTM都是transform上一个LSTM的输出。
* 在第四层LSTM的输出接了softmax预测word。

### Evaluation

* 主要在检索和生成任务中retrieval and generation tasks。

* Flickr30k和COCO2014。
* 在retrieval任务中使用median rank $Medr$，of the first retrieved ground truth image or caption and Recall @K，the number of images or vaptions for which a correct caption or image is retrieved within the top K results.
* 在generation任务中使用BLEU，AMT。

## Video description

作者设计了几种不同的结构，主要为LSTM和CRF：

* LSTM encoder，decoder with CRF max：通过使用maximum a posterior estimate（MAP） of a CRF taking in video features as unaries来提取语意信息，CRF会输出一些词语传递给LSTM，最终生成完整的句子。
* LSTM decoder with CRF max：CRF输出one hot，LSTM直接对输出decoder。
* LSTM decoder with CRF prob：CRF输出概率不经过max，LSTM直接对概率decoder。

### Evaluation

* TACoS。
* LSTM比SMT要好。
* b和c型都要比a型好。
* c型最好。







