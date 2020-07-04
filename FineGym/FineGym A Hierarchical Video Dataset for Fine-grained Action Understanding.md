# FineGym: A Hierarchical Video Dataset for Fine-grained Action Understanding

## 2020cvpr

> 主要贡献了一个详细标注的数据集。
>
> 在此之上测试了当前流行的方法，并作出分析。

## Abstract

* FineGym： a new dataset built on top of gymnastic videos.细粒度多层标注数据，分为event、set和element三层。

## Introduction

* 三条性质：
  * Multi-level：event, set, and element
  * Temporal structure：decomposed into sub-actions.
  * High quality
* 在FineGym数据上测试了多种模型，发现：
  * sparsely sampled frames are not sufﬁcient to represent action instances.
  * Motion information plays a signiﬁcantly important role, rather than visual appearance
  * Correct modeling of temporal dynamics is crucial.
  * And pre-training on datasets which target for coarse-grained action recognition is not always beneﬁcial.

## Related Work

#### Coarse-grained Datasets

* the background context often provides distinguishing signals, rather than the actions themselves.视频判别过程中背景很重要，甚至比运动本身更重要。

#### Methods for Action Recognition

methods could be summarized in three pipelines：

* 2D CNN to model per-frame semantics, followed by a 1D module to account for temporal aggregation.Specifically, TSN divides an action instance into multiple segments, representing the instance via a sparse sampling scheme. An average pooling operation is used to fuse perframe predictions. TRN and TSM  respectively replace the pooling operation with a temporal reasoning module and a temporal shifting module.
* 3D CNN to jointly capture spatial-temporal semantics, such as Nonlocal , C3D , and I3D .
* an intermediate representation (e.g. human skeleton in ) is used by several methods

其他与动作理解相关的任务还有

* action detection and localization
* action segmentation 
* action generation

## FineGym

#### Dataset Statistics

* 包括10个event categories：4个female和6个male events。共300多场比赛。其中对四种女子比赛（跳马、平衡木、自由体操和高低杠）进行了3+2的细致标注。

## Empirical Studies

#### Events/Sets

* TSN网络，发现在Events层RGB层占主导并且准确度基本饱和。而Sets层光流贡献更大。

#### Elements

涉及主流方法：

* 2D+1D模型，包括TSN, TRN, TSM和ActionVLAD；

* 基于3D卷积核的方法：I3D, Non-local;

* 最近火起来的基于人体关键点的识别方法，代表为ST-GCN。

#### 稀疏采样和密集采样

* 在UCF上只用2.7%的采样率（5帧）TSN的识别准确率就达到了饱和，而在FineGym上的元素类别识别则需要采样30%（12帧）以上的数据帧。

#### 时域信息的重要性

通过三方面印证：

* 对TSN而言，在给定不同的组类别进行元素类别识别时，光流信息相比于RGB特征对结果贡献显著更多
* TRN学习了帧间关系来建模时域信息，然而一旦将输入的帧的顺序打乱，TRN 的表现将大幅下降
* 对于没有时域建模的TSN来讲，当测试的帧数逐渐超过训练帧数，识别的表现会因为引入新信息而变好并饱和；而对于在模型设计中嵌入了时域建模的TSM来说，当测试帧数和训练帧数的差异过大，学到的时域模型不再适用，识别准确率将“一落千丈”

#### 模型预训练

* 一个可能的原因是细粒度与粗粒度动作的时域模式可能有较大的差异，因此预训练所学难以迁移。

#### 现有方法尚难解决的问题

- 密集、快速的运动，如各种空翻；
- 空间语义信息的细微差别，如腿部姿态的些微不同；
- 比较复杂的时域动态线索，如运动方向的变化；
- 基本的推理能力，如数出空翻的次数等。

