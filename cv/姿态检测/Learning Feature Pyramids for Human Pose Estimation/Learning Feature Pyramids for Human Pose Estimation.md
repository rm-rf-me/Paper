# Learning Feature Pyramids for Human Pose Estimation

### ICCV2017

## Abstract

人体姿态估计其中一项挑战是：摄像机镜头视角的变化和透视会对人体部位的比例产生明显的影响。特征金字塔的方法常在 DCNNs 中被用来处理推断时尺寸变换的问题（融合高层特征和低层特征）。作者进一步扩展该方法提出了金字塔残差模块（Pyramid Residual Module，PRMs）来增强 DCNNs 对尺度变化的鲁棒性。作者还拓展了当前的权重初始化方案拓展到多分支网络。

论文中的实验代码可在该地址获得：[https://github.com/bearpaw/PyraNet](https://github.com/bearpaw/PyraNet)

## 1 Introduction

**PRM** 

人体部位的准确定位是 CV 领域中有挑战性的基本任务，但是由于人体关节的高自由度、视角变化和透视，实现该任务很难。

DCNNs 已经能够根据人体大小将图像变形到相似尺度，再训练身体部位检测器 。尽管给出了正确的变换比例，但是由于不同人物图像的身体姿态，观察视角，关节运动不同，导致人体各部位的比例尺是不统一的。因此 DCNNs 对尺度变换具有不稳定性。

作者设计了*金字塔残差模块（PRM）*来增强 DCNNs 对于图像中尺度变化的鲁棒性。

> PRM 学习多种尺度下的特征卷积核，从而建立特征金字塔。步骤如下：
>
> * 给定输入特征，在多分支网络（multi-branch network）中，以不同的采样率进行下采样，来获得不同尺度下的特征
> * 使用卷积学习不同尺度的特征滤波器
> * 滤波后的特征被上采样到相同的分辨率，再把不同尺度的特征相加

*PRM 可以作为 DCNNs 结构的基础模块，在网络的不同层次学习特征金字塔。*

**分支网络权重初始化**

分支网络是网络设计的一个趋势 。所谓分支即某层的输入来多个层 or 层的输出被其他多个层使用。作者发现当前的权重初始化方案，如 MSR 和 Xavier 方法，不适用于分支网络，作者经过推导发现：网络参数的初始化应考虑分支数。

**激活方差累积问题**

作者还发现由于 identity mapping 导致残差单元的输出方差会随着深度的增加而积累 。

本篇论文的主要贡献有三点：

* 提出了 PRM，能通过学习 DCNNs 中的特征金字塔来增强深度模型对尺度的鲁棒性，且模型复杂度有很小的增加。
* 作者研究了有多个输入或输出分支的层的 DCNNs 初始化问题，提供了更好的权重初始化方案，方案可用于 Inception 和 ResNets 等模型
*  作者提出了由 identity mapping 引发的激活方差累积问题的解决方案

作者以 Hourglass network （conv-deconv）为网络的基本结构，实验本文的方法，达到 SOTA 结果

## 2 Related Work

## 3 Framework

<img src="https://i.loli.net/2020/08/27/m2bQRvJfKOZDpY5.png" alt="image-20200827153553489" style="zoom:50%;" />

### 3.1 Stacked Hourglass Network

Hourglass network 通过前馈的方式全面捕获信息，它首先通过对特征图进行下采样，执行自下而上的处理，然后通过对最底层的高分辨率特征进行合并来实现对特征图进行上采样，执行自下而上的操作。如上图（b）所示。自下而上，自上而下的过程重复多次，以构建 "stacked hourglass" 网络，并在每次堆叠的末尾进行中间监督。

基本的 hourglass 网络是由一个个的残差模块构建的，只能按照一比例来捕获特征。作者以金字塔残差模块（PRM）作为 hourglass 的组件，能够捕获多尺度的特征。 

### 3.2 Pyramid Residual Module（PRMs）

PRM 的目的是学习 DCNNs 的跨层次的特征金字塔。PRM 学习不同分辨率的输入特征的滤波器。另 $x^{(l)}$ 和 $W^{(l)}$  为第 $l$ 层的输入和滤波器，则 PRM 可以表示为：
$$
\mathbf{x}^{(l+1)}=\mathbf{x}^{(l)}+\mathcal{P}\left(\mathbf{x}^{(l)} ; \mathbf{W}^{(l)}\right)
$$
其中 $\mathcal{P}\left(\mathrm{x}^{(l)} ; \mathbf{W}^{(l)}\right)$ 是特征金字塔 ，表示如下：
$$
\mathcal{P}\left(\mathbf{x}^{(l)} ; \mathbf{W}^{(l)}\right)=g\left(\sum_{c=1}^{C} f_{c}\left(\mathbf{x}^{(l)} ; \mathbf{w}_{f_{c}}^{(l)}\right) ; \mathbf{w}_{g}^{(l)}\right)+f_{0}\left(\mathbf{x}^{(l)} ; \mathbf{w}_{f_{0}}^{(l)}\right)
$$
* $C$ 表示金字塔的层数

*  $f_{c}(\cdot)$ 是金字塔第 $c$ 层的变换公式

* $\mathbf{W}^{(l)}=$ $\left\{\mathbf{w}_{f_{c}}^{(l)}, \mathbf{w}_{g}^{(l)}\right\}_{c=0}^{C}$ 是参数集。


变换 $f_{c}(\cdot)$ 的输出相加，然后由滤波器 $g(\cdot)$ 进行卷积。下图显示了金字塔残差模块。为了减少计算和空间的复杂性，每个 $f_{c}(\cdot)$ 被设计为 bottleneck 结构。如下图所示，通过 $1 \times 1$ 卷积降低特征维度，然后通过 $3 \times 3$ 卷积在下采样得到的输入特征上计算新特征。 最后，所有新特征都将被上采样到相同的尺寸，再加在一起。

![image-20200827191643955](https://i.loli.net/2020/08/27/5GVvk4eOCfzoc6g.png)

**生成输入特征金字塔**：在 DCNNs 中一般应用 max-pooling 或 average-pooling 降低特征图的分辨率，编码平移不变性。 但是，pooling 因子至少为 2 ，会导致分辨率降低得太快，不能很好地生成金字塔。 为了获得不同分辨率的输入特征图，作者采用 fractional max-pooling 来近似用来生成传统图像金字塔的“平滑”和“下采样”过程。 第 $c$ 层金字塔的下采样率为：
$$
s_c = 2^{-M\frac{c}{C}},\quad c=0,……,C,M \ge 1
$$

* $s_c \in [2^{-M}, 1]$ 表示相对于输入特征图的分辨率
* 当 $c=0$  时，输出与输入分辨率相同；当 $M=1, c=C$ 时，输出分辨率是输入的一半。
* 实验中，作者将参数设置为 $ M = 1, C = 4$ ，使得金字塔的最小尺寸的输出分辨率是输入的一半。 

### 3.3 Discussion

1. PRM 对于 CNNs 是通用的。本文的 PRM 广泛应用于各种 CNNs 结构，如 Stacked Hourglass Network、Wide Residual Nets、ResNeXt.

2. 金字塔结构的变形：
   1. 上图 (a-b)，采用 fractional max-pooling + conv + upsampling 来学习特征金字塔
   2. 上图 (c)，采用 dilated conv 来学习特征金字塔
   3. 上图 (b) PRM-C，金字塔中不同层次特征的 summation 也可以替代为 concatenation
   4. 上图 (b) PRM-B，计算复杂度较小，同时能达到不错的性能

3. 权重共享。为生成特征金字塔，传统方法通常对不同层的图像金字塔学习，例如 HOG。 该过程对应于在金字塔的不同层 $f_c(\cdot)$上共享权重 $ W_{f_c}^{(l)} $ ，这能够大大减少参数的数量。

## 4 Training and Inference

作者使用 score map 来表示关节的位置。通过 $z = \{z_k\}_{k=1}^{K}$ 表示 ground-truth 位置，其中 $z_k =(x_k，y_k)$ 表示图像中第 $k$ 个关节的位置。 score map 的 ground-truth 为 $S_k$ 是均值为 $z_k$ 方差为 $\Sigma$ 的 Gaussian 分布如下：
$$
\mathbf{S}_{k}(\mathbf{p}) \sim \mathcal{N}\left(\mathbf{z}_{k}, \mathbf{\Sigma}\right)
$$
* $\mathrm{p} \in R^{2}$  表示位置
* $\Sigma$ 根据经验一般为单位矩阵
* 因为有 K 个身体关节，hourglass network 的每个堆栈，都给出 K个 score map，即 $\hat{\mathrm{S}}=\left\{\hat{\mathrm{S}}_{k}\right\}_{k=1}^{K}$ 。

对于 Loss 函数：hourglass network 的每个堆栈的末尾都附加了平方误差损失：
$$
\mathcal{L}=\frac{1}{2} \sum_{n=1}^{N} \sum_{k=1}^{K}\left\|\mathbf{S}_{k}-\hat{\mathbf{S}}_{k}\right\|^{2}
$$

* N 表示样本数。

对于最后一个 hourglass network 堆栈预测的 score map ，取 score map 的最大值位置作为关节的位置 $\hat z_k$ ：
$$
\hat z_k = argmax_p\hat S_k(p), k = 1, …, K
$$

### 4.1 Initialization Multi-Branch Networks

初始化对于训练非常深的网络是必不可少的，特别是对于密集预测的任务而言，由于归一化卷积网络的大量内存消耗导致较小的minibatch，因此 BN 的有效性较低。 现有的权重初始化方法是在假设无分支的纯网络的前提下设计的。PRM 具有多个分支，不符合假设。  因此，作者研究了如何初始化多分支网络。
$$
\operatorname{Var}\left[w^{(l)}\right]=\frac{1}{\alpha^{2}\left(C_{i}^{(l)} n_{i}^{(l)}+C_{o}^{(l)} n_{o}^{(l)}\right)}, \quad \forall l
$$

* $l$  表示层数
* $C_i^{(l)}$ 第 $l$ 层输入分支数
* $C_o^{(l)}$ 第 $l$ 层输出分支数

一般情况下。通常，具有分支的网络初始化参数时应考虑输入分支和输出分支的数量。 具体来说，如果将几个多分支层堆叠在一起而没有其他操作（例如BN，卷积，ReLU等），则使用 Xavier 或 MSR 初始化可以将输出方差提高大约$\prod _lC_i^{(l)}$ 倍。

###  4.2 Output Variance Accumulation

在多分支并行情况下，论文表明残差模块 identity mappings 恒等映射，使用 Xavier 和 MSR 初始化网络，方差成倍数增加。但是使用 BN+ReLU+conv 1$\times$1替换 identity mappings，就能避免方差爆炸。

## 5 Experiments

### 5.1 Experiments on Human Pose Estimation

1. benchmarks
   1. MPII 
   2. LSP

2. 实现细节

   输入的图像是根据标注的身体位置和比例从调整大小后的图像中裁剪的256×256。 对于LSP测试集，我们仅将图像中心用作身体位置，并根据图像尺寸估算身体比例。 通过缩放，旋转，翻转和添加色噪声对训练数据进行了扩展。 所有模型都使用 Torch 进行训练。 我们使用 RMSProp 在 4 个 Titan X GPU 上优化网络，minibatch 为 16个（每个GPU 4个），200 epoches。学习率初始化为 7×1e−4，在第 150 epoch 和 170 个 epoch 减小为 原来十分之一。 测试是通过翻转在6个尺度的图像金字塔上进行的。

3. 结果

   ![image-20200827233033947](https://i.loli.net/2020/08/27/OebIuURMmW95to8.png)

![image-20200827233135027](https://i.loli.net/2020/08/27/1YGVwXbKir49TeS.png)