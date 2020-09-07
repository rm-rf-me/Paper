# Human Pose Estimation with Spatial Contextual Information

### 2019没找到是哪个会议的。。。

### CPF，PGNN

## Abstract

* 大部分SOTA的姿态网络都是多阶段训练融合类似intermedia的多种deep supervision辅助。
* 作者提出了两个更简洁且易于计算的模型，命名为Cascade Prediction Fusion（CPF）和Pose Graph Neural Network（PGNN）。
* CPF模型通过融合之前阶段累积的预测map来抽取信息，融合后的map会被后续阶段用于预测。
* 为了提升关节点之间的空间关联，PGNN模型学习一种图形式的姿态结构信息，关节点之间能够互相传递信息从而表达出空间关系。
* 这两个网络需要的计算复杂度非常小。

## 1.Introduction

* 目前成功的网络都使用多个辅助的预测map，然后在多个阶段不断迭代和微调map直到产生最终结果。这就需要学到非常强的外形语意特征。并且要在训练时避免梯度消失。
* 为了很好的利用空间上下文信息，作者提出了两种模型。

#### 贡献1

* 为了利用上下文信息，作者提出了Cascade Prediction Fusion（CPF）模型来充分利用辅助预测map。之前阶段的预测map可以被看作是后续阶段的预测基础。
* 而CPF流程与Hourglass和CPM中是不太一样的。本文中的预测map会直接与图像特征拼接或者相加，然后传给后续的CNN块中。这是一种积累预测map的轻量级方法。
* 每个阶段都有着不同的特性，具体的，在较低层的网络中的预测map会更取决于局部信息，而在更高层的网络预测则会带有更强的语意信息，用于区分不同的关键点。CPF能通过更短的融合路径高效融合不同阶段的信息。

#### 贡献2

* 作者提出了PGNN，能够灵活而高效的学到关键点的结构化信息。本文的PGNN基于图结构，能够融入多种姿态检测网络中。
* 在图中的每个节点都与其相邻的关键点相连。空间信息通过边来表达。
* 与众不同的是，PGNN直接同时传递各个信息。不再是定义一种关节点的预测序列，PGNN能够动态的预测更新序列。
* 通过同时更新策略，能够得到长范围和短范围的关系信息。最终PGNN能够学到结构化的信息表达。
* 整个系统e2e。在MPII和LSP上用很高的计算效率达到了SOTA。

## 2.Related Work

### Human Pose Estimation

* 姿态检测的关键就在于对关键点的检测和空间信息的集成。之前的姿态检测方法可以被分为两部分：

  * 使用CNN学习特征表达。

    > * 最早在[DeepPose](Deeppose: Human pose estimation via deep neural networks. In CVPR, 2014.)中开始用多阶段模型。
    > * 之后[Fan](Combining local appearance and holistic view: Dual-source deep neural networks for human pose estimation. In CVPR, 2015.)等人用局部和全局特征提升了整体性能。
    > * 为了连接输入和输出空间，[Carreira](Human pose estimation with iterative error feedback. In CVPR,2016)在每次迭代步中拼接了前一阶段的预测map和输入图像。
    > * 借鉴了一些[语意分割的方法](Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. arXiv preprint arXiv:1606.00915, 2016.，Fully convolutional networks for semantic segmentation. In CVPR, 2015.)后，一些方法如[Hourglass](Stacked hourglass networks for human pose estimation. In ECCV, 2016.)、[CPM](Convolutional pose machines. In CVPR, 2016.)、[这篇](Learning feature pyramids for human pose estimation. In ICCV, 2017.)都会用高斯分布中心来表达每个部分的位置。

  * 建模各个关节点的空间位置关系。都是2011左右的paper，不说了。。。

### Graph Neural Network

* 图结构的特征学习工作同样可以分为两类：
  * 在图上应用CNN。基于图的拉普拉斯Graph Laplacian。为了能够直接在图上应用CNN，[这篇](Convolutional networks on graphs for learning molecular ﬁngerprints. In NIPS, 2015.)用了一种特殊的hash函数。
  * 另一类方法尝试在图中的每个节点使用recurrently applying neural networks。每个节点不同阶段的信息能够在边之间传递。
    * 图神经网络Graph Neural Network（GNN）在[这篇](The graph neural network model. IEEE Transactions on Neural Networks, 20(1):61–80, 2009.)中被首次提出。它利用多层感知机去学习图中每个节点的隐藏状态。而GNN的拓展，[Gated Graph Neural Network（GGNN）](Gated graph sequence neural networks. In ICLR, 2016.)使用了一种周期性recurrent的gating function来升级隐藏状态，并且输出序列。最终的参数可以通过back-propagation theough time（BPTT）算法更新。

## 3.Our Method

### 3.1.Cascade Prediction Fusion（CPF）

* 通常的目标检测方法都会在不同阶段产生预测图，不同级别的预测图所代表的内容是不同的。一般认为低维预测图负责局部信息，这在单个关节点的定位中非常重要。而高维预测图负责整体的语意信息。
* CPF就是用来融合各个维度级别语意信息的。它能够被应用到大多数的多阶段姿态检测框架中。
* 对于第i个阶段，不同于之前的方法会简单将当前阶段的prediction map：$pred_i$和前一个阶段的$pred_{i-1}$融合。CPF会将$pred_{i-1}$作为一个已知信息，参与当前$pred_i$的产生。具体的，前一阶段的较粗的$pred_{i-1}$会经过一个单核卷积增加通道数量，然后通过element-wise addition于图像特征合并。然后fused之后得到的map作为输入来生成$pred_i$。

### 3.2.Graph Neural Network（GNN）

* 图表示为$G=\{K, E\}$，其中的$K$为节点，$E$为边。对每个节点$k\in K$都关联着一个隐藏状态向量$h_k$，该向量是在每个周期不断更新的。在时间步$t$时表示为$h_k^t$。

* 每次更新该向量时，都会将当前向量和其相邻节点$\mathcal{N}_k$发来的信息$x^t_k$作为输入。$\mathcal A$是整合邻居信息的函数，$\mathcal T$为更新隐藏信息的函数。所以最终的更新结果为：
  $$
  x^t_k=\mathcal A \left ( h^{t-1}_u|u\in \mathcal N_k\right ) 
  $$

  $$
  h_k^t=\mathcal T\left ( h_k^{t-1}, x^t_k\right )
  $$

  

#### Graph Construction

* 在PGNN中的每个节点都代表了一个关节，每条边都代表了相邻关节之间的联系。关节图可以建成一棵树。

* 每个向量都可以初始化为通过原始图像抽取到的空间信息：
  $$
  h_k^0=\mathcal F_k\left ( \Theta, I \right ), k\in \{1...K\}
  $$
  其中的$\mathcal F$代表backbone network，$\Theta$代表网络的参数，$I$为原始输入的图像。

#### Information Propagation

* 作者使用构造好的图来学习空间关联信息并且在每个阶段中不断微调各个关节的表达。在更新每个节点的hidden state之前，首先会Aggregates整合上一阶段各个相邻节点的hidden state信息。

* CNN层能够很好的对集合信息进行transform。这里注意不同边的Conv参数是不共享的。所以$\mathcal A$就被表示为：
  $$
  x^t_k=\sum_{k, k'\in \Omega}W_{p, k}h_{k'}^{t-1}+b_{p, k}
  $$
  其中的$W_{p, k}$为卷积权重b为偏置。$\Omega$为相连的点的集合。

* 而对于$\mathcal T$：

  $z_{k}^{t}=\sigma\left(W_{z, k} x_{k}^{t}+U_{z, k} h_{k}^{t-1}+b_{z, k}\right)$
  $r_{k}^{t}=\sigma\left(W_{r, k} x_{k}^{t}+U_{r, k} h_{k}^{t-1}+b_{r, k}\right)$
  $\tilde{h}_{k}^{t}=\tanh \left(W_{h, k} x_{k}^{t}+U_{h, k}\left(r_{k}^{t} \odot h_{k}^{t-1}\right)+b_{h, k}\right)$
  $h_{k}^{t}=\left(1-z_{k}^{t}\right) \odot h_{k}^{t-1}+z_{k}^{t} \odot \tilde{h}_{k}^{t}$

  公式的思想借鉴了GRU的设置。引入了门控的思想。

#### Output and Learning

* 在经过了T步更新之后，我们最终得到了预测：
  $$
  \tilde P_k=h_k^T+h^0_k
  $$
  即为融合第T步得到的预测和初始化时的关节点外形特征。

* 最终的图网络使用最小化L2损失训练：
  $$
  L_{2}=\frac{1}{K} \sum_{k=1}^{K} \sum_{x, y}\left\|\widetilde{P}_{k}(x, y)-P_{k}(x, y)\right\|^{2}
  $$

#### 3.2.1 Graph Types

* 理想情况下，全连接图应该能够根据其他关节来收集信息的。但是有些关节，比如头和脚踝就很难捕捉到这种关系。
* 为了解决这个问题，作者使用了两种结构：tree结构和loopy结构。树结构非常简单，能够收集到紧邻关节间的关系信息。而Loopy结构则更加复杂，该结构允许信息在其上的loop上传递。

#### 3.2.2 Relationship to Other Methods

* 大部分的SOTA更关注学习关节点的外形，他们通过扩大感受野的方法来获得关节点之间的关系。但是，人体的姿态可能差距会非常大，使用结构化信息更能够促进效果的发挥。
* 其他的方法如RNN和Probabilistic Graphical Model（PGM）也能够建模到关系信息。

##### PGNN vs RNN

* RNN可以被看作一种特殊的PGNN。它也能够在图中的各个节点之间传递信息，其中图上的节点就是各个关节点，关节间关系通过边来传递。
* 在RNN中，每个时间步都会根据当前state和上阶段的hidden state来更新节点。这和PGNN有所不同，PGNN会从各个邻居处收集信息。
* 还有就是RNN的输入必须手动定义，任何不老合适的设置都会摧毁关节点之间的关系结构。

##### PGNN vs PGM

### 3.3.Backbone Networks

作者使用了两种backbone：ResNet-50和8-stack Hourglass

#### 3.3.1 ResNet-50

* 首先移除ResNet最后的classification和average pooling。然后融合进了CPF。最终使用PGNN提升结果。

##### Feature Pyramid Network

* 在ResNet中引入了FPN，并同样使用了lateral connection来整合bottom-up和top-down的信息。最终在三个不同的层次得到了三个辅助的预测结果。

##### Dilated Convolution（空洞卷积）

* [Dilated Convolution](Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. arXiv preprint arXiv:1606.00915, 2016.)能够在不增加参数量的同时提升模型感受野。一张输入的图像在经过ResNet-50之后会被down-sampled 32次。但是这样的特征图对于关节分类来讲还太粗糙。
* 为了解决这个问题，作者首先将最后两个卷积层的步长从2px减少到1px。这减少了感受野。而在CPM和Hourglass中空间信息都需要足够大的感受野，所以作者用dilated卷积代替了最后的两个3*3卷积。

##### Other Implementation Details

* ImageNet预训练。
* RMSProp优化器。
* 250 epochs，batch size 8.
* lr=0.001。在200轮之后除10.

#### 3.3.2 Hourglass

##### Implementation Details

* RMSProp。
* 参数随机初始化。
* 300 epoch，batch size = 6.
* lr = 0.00025，240轮之后除10.

## 4.Experiments

### Datasets

* MPII和LSP。

### Data Augmentation

* cropped and warped to 256*256。
* scaling [0.75,1.25]
* rotation -30~+30
* horizontal flipping

### Unary Maps

* 用最后一层的预测作为unary map，因为此处的预测综合了全部的语意信息。
* 最后的score map为C\*64\*64，是原图的1/8.

### Evaluation Criteria

* LSP用Percentage Correct Keypoints（PCK），MPII用PCKh。

### 4.1 Ablation Study

#### CPF

* 在ResNet50上有CPF和没有CPF。较难的关节大概能提升个一个点左右。

#### PGNN

* 剩下的就是说明文了，不翻译了。。。

