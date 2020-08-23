# Convolutional Pose Machines

### CVPR 2016 CPM

### [代码开源](https://github.com/CMU-Perceptual-Computing-Lab/convolutional-pose-machines-release)

这篇paper看的时候最好先结合3.1去了解一下pose machines，因为整体模型还是非常机器学习的。然后就是这个作者非常喜欢写巨长的句子。。。这对我这个英语菜狗很不友好。。。

整篇看了一遍之后很多细节还不是很清楚，所以文档写的挺烂的，好在是代码开源所以等我补一篇代码注释吧。。。





## Absrtact

* Pose Machines提供了一套用于预测丰富空间信息的框架，在本论文的工作中，我们将卷积网络集成进PM框架中来学习图片特征和图片空间模型并用于人体姿态预测。
* 本论文的主要贡献就在于提出的模型能够清晰的学习到多个变量之间的空间关联关系，因此该模型十分适用于去解决类似姿态关键点预测的问题。
* 我们提出的其实就是一个包含了多个卷积网络的序列框架，该架构使得网络能够直接用上一阶段网络输出的置信图来进一步精确预测每一个关键点的位置。
* 该网络架构的提出，还天然的用到了中继监督训练的方法（intermediate supervision），而这种训练方法经过实验论证也确实能够在一定程度上消除梯度消失问题。我们在多个数据集拿到sota，包括MPII, LSP和FLIC。

## 1.Introduction

* CPM模型继承了[pose machine](Pose Machines: Articulated Pose Estimation via Inference Machines)的优点：能隐式学到图像和multi-part cues的long-range依赖。对于pose machine，这里放一个湾湾的[笔记]([https://medium.com/@pomelyu5199/%E8%AB%96%E6%96%87%E7%AD%86%E8%A8%98-eccv14-pose-machines-8c8f53c2f72a](https://medium.com/@pomelyu5199/論文筆記-eccv14-pose-machines-8c8f53c2f72a))。同时CPM还引入了CNN的优点，如同时学习图像和空间关系特征的能力。该模型能够e2e训练，并且能够处理大的数据。
* CMP在每个部分都会产生2D belief maps。在CPM的每个阶段都使用前一阶段产生的图像特征和belief map作为输入。
* 其中的belief maps传递给后续阶段的对每个空间相对关系的编码信息，这是的CPM能够学到不同部分之间丰富的依赖关系。而且不同于传统的graphical模型合作specialized post-processing strps，本文使用CNN来生成belidf maps。从而整个多阶段模型变得非常不同，并且能够e2e训练。
* 在CPM中的每个阶段，part beliefs中的空间上下文关系能给后续阶段的消除歧义提供有力支持。模型随着阶段的推进逐渐对各个部分更细致的区分。未来能够得到long-range的影响，各个阶段的目标都是达到a large receptive field on both the image and the belief maps。在实验中发现，large receptive field的beliefs map在大范围空间关系的表达和最终acc中至关重要。
* 深层网络可能会出现梯度消失的问题。近期的研究表明，监督深层网络的中间部分对训练是有帮助的。本文将展示如何在CPM中避免这个问题。此外，还将讨论该序列预测结构的不同训练模式。
* 作者的主要贡献为：
  * 通过融合了CNN的序列模型学到了内在的空间关系。
  * 设计和训练这样的能够既抽取图像特征又抽取空间关系的的结果预测模型的方法，同时这东西不需要任何的graphical model style inference。
* 模型在MPII，LSP和FLIC上达到sota

## 2.Related Work

* 传统的pose estimation方法是pictorial structures模型。其中的不同的身体结构被表达进树结构中。
* Hierarchical models在树中表达了不同尺寸的各个部分的关系。这些模型都建立在一个共同的假设上，即人的肢体部分都表现在图像中并能够较容易地定位，从而通过这些信息帮助检测那些较难识别的部分。
* Non-tree。
* Sequential prediction
* Convolutional architectures

## 3.Method

### 3.1 Pose Machines

* 定义身体某个部位的关键点坐标为$Y_{p} \in \mathcal{Z} \subset \mathbb{R}^{2}$,其中$\mathcal Z$为全体像素点的集合。而我们的目标就是预测出所有P个关键点的坐标。

* Pose Machine包括一系列的multi-class predictors $g_t(·)$，被训练来预测每一个关键点的位置。对于每个阶段$t\in \{1...T\}$，分类器$g_t$对于$Y_p=z,\forall z \in \mathcal{Z}$，都会根据图像的特征和同阶段的$Y_p$周围的分类器的信息来预测置信度。

* 对于t=1即第一层的分类器来说，构造这样的置信度：
  $$
  g_1(\mathbf{x}_z) \rightarrow \{b_1^p(Y_p=z)\}_{p\in \{0...P\}}
  $$
  其中的$b_1^p(Y_p=z)$为$g_1$的分数。

* 这里设定第p个关键点在图像中每个位置$z=(u, v)^T$的置信为$\mathbf{b}^p_t\in \mathbb{R}^{w\times h}$。这里的h和w为图像的高和宽，具体的：
  $$
  \mathbf{b}^p_t[u, v]=b^p_t(Y_p=z)
  $$
  为了方便，直接将P个关节点和1个背景表示为：
  $$
  \mathbf{b}_t\in \mathbb{R}^{w\times h \times (P+1)}
  $$

* 在后面的阶段中，分类器判断置信值主要依靠两个信息：

  * 图像信息：$\mathbf{x}_z^t \in \mathbb{R}^d$

  * 相邻分类器的信息

  * 即：
    $$
    g_t(\mathbf{x}'_z, \psi_t(z, \mathbf{b}_{t-1})) \rightarrow \{b^p_t(Y_p=z)\}_{p\in \{0...P\}}
    $$
    其中的$\psi_{t>1}(·)$为$\mathbf b_{t-1}$到上下文特征的映。

* 对于每个阶段，计算的置信对每个关节点的估计都越来越精细。
* 注意这里的图像特征$\mathbf x'_z$在后续阶段是可以和第一阶段不同的。在Pose Machine中使用了boosted random forests作为分类器$g_t$，手工设计每个阶段的$x'=x$，手工设计特征图$\psi_t(·)$来抽取每个阶段的空间上下文信息。

### 3.2 Convolutional Pose Machines

#### 3.2.1 Keypoint Localization Using Local Image Evidence

* 第一阶段的CPM仅依靠局部图像信息local image evidence来预测关键点置信。local是因为第一阶段的receptive field仅是输出像素周围的一小个patch。使用的CNN有五个卷积层随后还有两个$1\times1$卷积。
* 在实验中，作者将输入normalize到$368\times368$，接受野receptive field为$160\times160$。
* 整个第一阶段网络可以看作是使用CNN结构在整张图像上滑动，然后根据160*160的每个patch回归出P+1这个确定尺寸的输出向量，代表了在每个像素位置的每个关节点的置信分数。

#### 3.2.2 Sequential Prediction with Learned Spatial Context Features

* 虽然具有一致外观的地标（例如头和肩膀）的检测率可能是有利的，但由于其轮廓和外观存在较大差异，对于人体骨骼运动链较低的地标，其精度通常要低得多。 但是，尽管有噪声，但信念图的周围部分位置可能会提供很多信息。就比如说右肩膀的置信位置对右手肘的判定意义重大。
* 在t>1阶段中的分类器能够在有噪声的belief maps中考虑对于当前像素周围的空间上下文$\psi_{t>1}(·)$并借此提升模型效果。比如第二阶段的分类器$g_2$会接受两个输入：特想特征$x^2_z$和通过$\psi$计算前一阶段beliefs得到的特征。
* 此处的特征函数$\psi$代表了对前一阶段不同关节点当前像素周围的空间信息。CPM 没有显式函数来计算空间内容特征，而是，定义特征函数作为分类器在先前 stage 的 beliefs 上的接受野(receptive field).
* 网络设计的原则是：在 stage t=2t=2 网络的输出层的接受野是足够大的，以能学习不同关节点间的复杂和long-range关联性。为了能简洁的抽取出之前阶段输出中的特征，后序序列的CNN层循序分类器自由的结合上下文信息by picking the most predictive features。
* 第一阶段的置信map仅根据使用小接受野的的网络生成。而第二阶段中，接受野极速增加。大的感受野能够通过几种种方式获得：
  * pooling，代价是损失精度precision
  * 增加卷积核的尺寸，代价是参数激增。
  * 加深层数，代价是梯度消失。
* 总的来说第二阶段为把以下三个数据合一：
  \- 阶段性的卷积结果（46*46*32） 纹理特征
  \- 前一阶段各部件响应（46*46*10） 空间特征
  \- 中心约束（46*46*1）
  串联后的结果尺寸不变，深度变为32+10+1 = 43。
* 本文中作者使用多层CNN在$8\times$downscaled heatmaps上实现大的感受野，此处的8是指原图尺寸为386\*386缩放了八倍之后的feature map大小为46*46，这也控制了模型的参数。作者发现，stride-8的网络效果核stride-4的效果差不多。
* 作者还发现不同感受野带来的效果是不同的（当然），对于304*304尺寸的图像来说，acc随着感受野上升，并在250左右开始收敛。这个在acc的提升代表了网络本身并没有成功编码各个关节间的long range关系。

### 3.3 Learning in Convolutional Pose Machines

* 上述的网络层数可能会非常深，不得不直面梯度消失的问题。

* 幸运的是，pose machine的序列预测框架提供了一种解决方法。对于每个阶段的pose machine都在不停的产生着每个位置每个关节的belief maps。

* 作者在阶段t定义了一个loss function，目标是最小化预测和ideal belief maps之间的L2距离。其中对于每个关节p的ideal belidf map写为：$b^p_*(Y_p = z)$，是通过在每个关节的ground truth位置放置Gaussian peaks得到的。在每个阶段中希望最小化的代价函数为：
  $$
  f_{t}=\sum_{p=1}^{P+1} \sum_{z \in \mathcal{Z}}\left\|b_{t}^{p}(z)-b_{*}^{p}(z)\right\|_{2}^{2}
  $$
  最终整体的loss即为：
  $$
  \mathcal{F}=\sum_{t=1}^{T} f_{t}
  $$

* 作者使用SGD联合训练所有T个阶段，为了在所有后续阶段中共享图像特征，作者共享了对应的CNN参数。

## 4.Evaluation

### 4.1 Analysis

#### Addressing vanishing gradients

* 上面定义了一个decomposable的损失函数，由各个部分组成。这个操作叫做中继监督优化Intermediate supervision，如果直接对整个网络进行梯度下降，输出层的误差经过多层反向传播会大幅减小，即发生vanishing gradients现象。为解决此问题，在每个阶段的输出上都计算损失。这种方法可以保证底层参数正常更新。该方法能够有效解决深层网络的梯度消失问题，因为中继层的损失函数能够给每个阶段及时的梯度补充。
* 作者检测了使用和不使用该方法的模型训练中各个层在训练过程中的梯度大小，发现在不使用时前几层梯度几乎始终为0，而后几层则更大，意味着模型退化为后几层。而使用了之后梯度分布就很均匀。 

#### Benefit of end-to-end learning

* 有了CNN在老方法上提了三四十个点。

#### Comparison on training schemes

* 为了证明Intermediate supervision的效果，作者使用四种方法训练网络：
  * 1.使用全局损失功能从零开始进行培训，以加强中间监督。
  * 2.stage-wise：where each stage is trained in a feed-forward fashion and stacked
  * 3.和1.相似但是使用2.来初始化参数权重
  * 4.和1.相同但是么得intermediate supervision。
* 结果为第一种最好，第二种最差，第三种因为只用了第一种的方法做finetune，所以将第二种的结果拉高至非常接近最好。

#### Performance across stages

* 阶段越多效果越好。

### 4.2 Datasets and Quantitative Analysis

* 这部分测试模型在多个benchmark中的表现：MPII、LSP、FLIC。
* 为了normalized输入到368*368，首先缩放所有图片到相同的尺寸，然后corp或者pad这些图像的中心位置。
* [代码开源](https://github.com/CMU-Perceptual-Computing-Lab/convolutional-pose-machines-release)

#### MPII

* 数据增强：
  * 随机左右旋转40度
  * 图片缩放0.7-1.3
  * 水平翻转
* 评测方法：
  * PCKh metric
* PCKh-0.5 SOTA，提升了6个点。其中最难的脚踝关节提升了十个点。

#### LSP

* person-centric标注。
* 评测方法：
  * Percentage Correct Keypoints（PCK）metric
* 相同的数据增强方法。
* 模型同样SOTA。
* 注意到如果融合经MPII数据模型效果能提升五个点。这是因为MPII有更高质量的标注。

#### FLIC

* 除了良好的整体效果外，更亮眼的是对困难关节的巨大提升，在wrists和elbow能有十个点左右的提升。