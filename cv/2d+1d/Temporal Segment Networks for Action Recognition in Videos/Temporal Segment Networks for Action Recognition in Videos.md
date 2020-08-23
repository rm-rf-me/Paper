# Temporal Segment Networks for Action Recognition in Videos

### TSN IEEE 2018

## Abstract

* We present a general and ﬂexible video-level framework called temporal segment network (TSN), aims to model long-range temporal structures with a new segment-based sampling and aggregation module. 
* 模型使用simple average pooling and multi-scale temporal window integration可以适用于修剪和未修剪的数据上。
* TSN在四大数据上SOTA。

***

## Introduction

* 视频动作检测有两点至关重要：appearances和temporal dynamics。

* 系统的检测性能受到能否抽取并利用相关信息。而此步骤也受到尺寸、焦点改变和镜头移动等限制。

* 深层CNN近年来发展迅速，也涉及在video方面的方法。但相比于传统手工抽取特征的传统action recognition方法还有差距。

* 作者认为ConvNet在video action中应用的三个阻碍：

  * 在传统方法对内容的语义理解中长范围时序结构已经被证明起到至关重呀的作用，而ConvNet则更偏向于段范围信息。一些解决方法依靠大密度抽样，但这样会增大计算开销并且限制模型的接受规模。
  * 现行方法大多只针对剪辑视频。而未剪辑视频中可能充斥着大量背景信息作为干扰。
  * 实际训练数据受限，过拟合严重。同时短期的光流抽取成为large-scale动作检测的瓶颈

  这明确了三个研究目标：

  * 如何高效的抽取长范围时序信息。
  * 如何把ConvNet网络部署到未剪辑视频中。
  * 如何高效的利用小数据训练网络并使其在大数据中良好泛化。

  作者给出的解决方法为：

  * 对于长范围时序结构，提出了temporal segment netword（TSN）其核心为：

    >  consecutive frames are highly redundant, where a sparse and global temporal sampling strategy would be more favorable and efﬁcient in this case.

    TSN的抽帧策略为先对长的视频序列进行分段，然后每个分段中随机选取一个短片段snippets，接着使用segmental consensus函数来聚合抽样的snippets。经过实验得到了五种聚合函数aggregation functions，包括三种基础形势：average pooling、max pooling、weighted average，和两种进阶模式：top-K pooling和adaptive attention weighting。其中后两个是为了在训练中自动标记出决定性的snippets，减少不相干snippets的影响。

  * 为了将TSN部署到未剪辑的视频中，作者设计了分级聚合的策略hierarchical aggregating strategy，取名为Multi-scale Temporal Window Integration（M-TWI）。首先将未剪辑的长视频切分成各个短窗口序列，然后独立地对每个窗口做action recognition，其中包括max pooling窗口内的snippet-level的recognition分数。最后使用top-K pooling或attention weighting聚合每个窗口得到整个视频的recognition结果。

  * 为了解决模型训练和部署的问题，作者发现通过限制训练样本，并且在输入上使用systematicl study能够释放ConvNet的潜力。特别的，作者提出了cross-modality initialization策略来把RGB模型的representations迁移到其他模型比如optical flow或者RGB difference。其次，作者创造了一种用于fine-tuning scenario的Batch Normalization，叫做partial BN。还有，为了更丰富的利用信息，模型采用四种输入：single RGB,stacked RGB difference, stacked optical flow field, stacked warped optical flow field。

  该模型在四个数据上达到了SOTA。并且将action recognition网络化成了最新的模型，比如ResNet和Inception V3，同时将音频信息也纳入辅助通道中。

***

## Related Work

#### Video Representation

可以粗略的分为两类：人工抽取特征和深度学习抽取特征。

##### Hand-crafted features

* 朴素的时空特征抽取方法非常多样，但These local features share the merits of locality and simplicity,容易缺失语意信息和判别能力。解决方法也很多，不一一列举了。。。总之就是这些方法具有了语意表达能力和判别能力，但依旧没有摆脱底层手工抽取的信息。

##### Deeply-learned features

* 之前的诸多工作，包括双流、3D卷积、分解卷积和新的3D卷积decompose方法、轨迹池化卷积TDD、RNN等。

#### Temporal Structure Modeling

* 时序结构有很多中，比如对每个原子动作做注释的ASM模型、使用潜在变量来建模复杂动作分解并使用Latent SVM学习模型参数、使用Latent Hierarchical模型和Segmental Grammar模型解耦复杂动作、序列骨架模型SSM、BoVW等。
* 这些模型都尝试建模长序列信息，但都使用连续帧序列接RNN或3D CNN。然而受限于GPU内存，一般只能覆盖64-120帧的信息。
* 而本模型不一样，本模型选用离散的抽帧策略，极大的减少了计算开销，增大了感受范围。

***

## Temporal Segment Networks

这一部分将详细讨论整个模型，分为以下四个部分：segment based抽样的动机、temporal segment网络结构、多种聚合函数和分析、问题分析。

### Motivation of Segment Based Sampling

* 之前的无论是双流还是3D卷积都是针对一帧或者很有限的几帧来做文章，根本无法把控全局的信息。
* 一些解决办法要不是增大抽帧数量。要不是使用固定步长抽帧。而这些方法在计算量和模型表达上都不占优，计算上密集抽帧显然不合适。而模型上步场抽帧终究还是个局部信息，也只能覆盖几秒钟的内容。
* 作者观察到虽然帧数在视频中非常密集，但是信息变化得极其缓慢，进而提出了segment based sampling方法。这是一种全局离散的方法，

### Framework and Formulation

* 作者设计的TSN模型工作在segment based抽样在全局视频中抽出的一坨小的片段snippets。

* 为了让这些snippets既能够保留全局信息又有很小的计算开销。segment based sampling首先将长视频切分为定长的小段segments，然后在每个segment中随机抽一个snippet。每个snippet都计算其自身的类别分数，然后使用一致性函数consensus function聚合各个snippet-level的预测分数。

* 公式为：

  $$\begin{array}{l}\operatorname{TSN}\left(T_{1}, T_{2}, \cdots, T_{K}\right)= \mathcal{H}\left(\mathcal{G}\left(\mathcal{F}\left(T_{1} ; \mathbf{W}\right), \mathcal{F}\left(T_{2} ; \mathbf{W}\right), \cdots, \mathcal{F}\left(T_{K} ; \mathbf{W}\right)\right)\right)\end{array}$$

  其中的F为snippet的分数函数，G为聚合函数，H为输出函数。这里H直接是Softmax。

* 融合了分类的交叉熵和聚合函数，最终的loss为：

  $$\mathcal{L}(y, \mathbf{G})=-\sum_{i=1}^{C} y_{i}\left(g_{i}-\log \sum_{j=1}^{C} \exp g_{j}\right)$$

  其中C为分类数量，$y_i$为groundtruth，$g_i$为G的第$j^{th}$维。

* 求导为：

  $$\frac{\partial \mathcal{L}(y, \mathbf{G})}{\partial \mathbf{W}}=\frac{\partial \mathcal{L}}{\partial \mathbf{G}} \sum_{k=1}^{K} \frac{\partial \mathbf{G}}{\partial \mathcal{F}\left(T_{k}\right)} \frac{\partial \mathcal{F}\left(T_{k}\right)}{\partial \mathbf{W}}$$

  其中K为segments的数量。作者使用了SGD。

### Aggregation Function and Analysis

* 聚合函数是模型中最重要的部分。
* 作者提出了五种函数：max pooling, average pooling, top-K pooling, weighted average, attention weighting。

##### Max pooling

* 公式为：$g_{i}=\max _{k \in\{1,2, \cdots, K\}} f_{i}^{k}$,其中$f_{i}^{k}$为$\mathbf{F}^{k}=\mathcal{F}\left(T_{k} ; \mathbf{W}\right)$的第i个元素。从而倒数为：

  $$\frac{\partial g_{i}}{\partial f_{i}^{k}}=\left\{\begin{array}{l}1, \text { if } k=\arg \max _{l} f_{i}^{l} \\ 0, \text { otherwise }\end{array}\right.$$

* 这个思路最直接的目的是对每一个动作类别寻找一个唯一的最具有决定性的snippet。这个思路最直接，但显然直接丢掉了其他的所有信息，这就很不合适了。

##### Average pooling

* 对max pooling最直接的改进就是使用均值：$g_{i}=\frac{1}{K} \sum_{k=1}^{K} f_{i}^{k}$,从而导数就成为了

  $\frac{\partial g_{i}}{\partial f_{i}^{k}}=\frac{1}{K}$

* 这样平均下来能够综合考虑全局信息。但是在一些噪声较大的数据中，一些错误的snippet会破坏最终的结果。

##### Top-K pooling

* 为了平衡最值和均值，作者创造了top-K方法。对前K个关键snippet做均值：$g_{i}=\frac{1}{\mathcal{K}} \sum_{k=1}^{\mathcal{K}} \alpha_{k} f_{i}^{k}$，其中$\alpha_{k}$为值为0或1的选择参数。所以最终的导数为：

  $\frac{\partial g_{i}}{\partial f_{i}^{k}}=\left\{\begin{array}{l}\frac{1}{\mathcal{K}}, \text { if } \alpha_{k}=1 \\ 0, \text { otherwise }\end{array}\right.$

* 这里的K选在1和全部段数K之间。这个方法一看就很雨露均沾（~~实属渣男行为~~）。

##### Liner weighting

* 在这个函数中作者想要表现出每个动作预测分数在element-level的线性组合特性。所以定义：$g_{i}=\sum_{k=1}^{K} \omega_{k} f_{i}^{k}$,从而得到：

  $\frac{\partial g_{i}}{\partial f_{i}^{k}}=\omega_{k}, \quad \frac{\partial g_{i}}{\partial \omega_{k}}=f_{i}^{k}$

* 这样的设置能够更加细致的拟合出不同动作的组合情况。相比于之前的pooling来讲这个更加soft。

##### Attention weighting

* 显然上面的线性方法跟具体的数据有很大关系，不能够很好的考虑到不同数据间的差异。所以作者提出了attention weighting。作者希望训练一个函数能够自动分配视频中不同snippet的比重。

* 所以定义：$g_{i}=\sum_{k=1}^{K} \mathcal{A}\left(T_{k}\right) f_{i}^{k}$， 其中$\mathcal{A}\left(T_{k}\right)$为snippet $T_k$的attention权重。从而导数为：

  $\frac{\partial g_{i}}{\partial f_{i}^{k}}=\mathcal{A}\left(T_{k}\right), \quad \frac{\partial g_{i}}{\partial \mathcal{A}\left(T_{k}\right)}=f_{i}^{k}$

* 在attention中，A函数对最终的结果非常重要。经过作者的实践，首先对每个snippet使用相同的ConvNet抽取特征$\mathbf{R}=\mathcal{R}\left(T_{k}\right)$,然后添加权值和softmax：

  $\begin{array}{c}e_{k}=\omega^{a t t} \mathcal{R}\left(T_{k}\right) \\ \mathcal{A}\left(T_{k}\right)=\frac{\exp \left(e_{k}\right)}{\sum_{l=1}^{K} \exp \left(e_{l}\right)}\end{array}$

* 所以得到导数为：

  $$
  \frac{\partial \mathcal{A}\left(T_{k}\right)}{\partial \omega^{a t t}}=\sum_{l=1}^{K} \frac{\partial \mathcal{A}\left(T_{k}\right)}{\partial e_{l}} \mathcal{R}\left(T_{l}\right)
  $$

  其中：

  $$
  \frac{\partial \mathcal{A}\left(T_{k}\right)}{\partial e_{l}}=\left\{\begin{array}{l}\mathcal{A}\left(T_{k}\right)\left(1-\mathcal{A}\left(T_{l}\right)\right), \text { if } l=k \\ -\mathcal{A}\left(T_{k}\right) \mathcal{A}\left(T_{l}\right), \text { otherwise }\end{array}\right.
  $$

* 所以全局的损失函数为：
  $$
  \frac{\partial \mathcal{L}(y, \mathbf{G})}{\partial \mathbf{W}}=\frac{\partial \mathcal{L}}{\partial \mathbf{G}} \sum_{k=1}^{K}\left(\frac{\partial \mathbf{G}}{\partial \mathcal{F}\left(T_{k}\right)} \frac{\partial \mathcal{F}\left(T_{k}\right)}{\partial \mathbf{W}}+\frac{\partial \mathbf{G}}{\partial \mathcal{A}\left(T_{k}\right)} \frac{\partial \mathcal{A}\left(T_{k}\right)}{\partial \mathbf{W}}\right)
  $$

* 总的来说，attention带来的好处为：
  * 模型自主选择权重，强化了模型的表达能力。
  * attention基于ConvNet。这能增加额外的反向传播信息，有助于收敛。

### TSN in Practice

##### TSN Architectures

* 在研究过程中使用了Inception V2。而在实际比赛中，模型使用了Inception V3和ResNet152这样的大杀器。

##### TSN Inputs

* 除了RGB和optical flow，作者还使用了另外两种输入：
  * Warped Optical Flow
  * RGB Differences：光流抽取是双流网络的速度瓶颈之一。所以作者提出了一种不需要光流的动作表达方式。受其他研究启发，直接使用RGB differences也能够表达出运动信息。

##### TSN Training

正如上文提及的，当前的数据集数量有限，过拟合现象严重。为了避免这一问题，作者使用了多种方法。

* Cross Modality Initialization：当训练数据较小时先行在大数据上做pre train是一个非常有效的做法。对于空间的RGB部分直接做pre train。而对于时间的flow和RGB差部分作者采用了cross modality initialization策略。首先将光流域线性离散化到0～255区间，然后对pretrained RGB模型参数根据第一层的通道数取均值，最后剩余层直接copied pretrained的模型。
* Regularization：正则化能够加速模型收敛，但同时增加了过拟合的风险。所以在导入pretrain模型之后，作者选择固定除了第一层外的BN的mean和variance参数。因为光流分布和RGB分布不同，所以需要第一层来重新拟合分布。称这种方法为部分partial BN。同时作者在全局pooling之后添加了额外的高比率的dropout层（0.8）来防止过拟合。
* Data Augmentation：在传统双流法中就使用了随机复制和水平翻折。作者又提出了两种新的数据增强方法：corner cropping和scale-jittering。
  * corner cropping：抽取的regions来自角落和中心，这是为了防止模型过度聚焦在中心区域上。
  * Multi-scale cropping：作者将输入尺寸固定到256$*$340上，并且cropped regions的宽和高从256、224、192、168中随机选择，最后这些regions被缩放到224$*$224上作为训练输入。实际上这不止是尺寸压缩，还是比例压缩。

***

## Action Recognition with TSN Models

### Action Recognition in Trimmed Video

* 剪辑视频就是分类任务，没啥内容不说了。。。

### Action Recognition in Untrimmed Videos

* 未剪辑视频最大的障碍就是充满大量的冗余信息。如果想要把在trimmed数据上训练的模型部署到untrimmed数据上，需要解决一些问题：
  * Location issue：动作的开始位置不确定。
  * Duration issue：动作的持续时间不确定。
  * Background issue：无用信息非常多并且可能会占用非常大的比例。
* 为了解决这些问题，作者设计了detection based方法。首先，为了覆盖可能出现在任何位置的动作，作者使用固定频率抽样snippets（1FPS）。然后，为了覆盖含有大变化的动作区间，一系列不同尺寸的temporal sliding窗口将在frame scores上滑动，每个窗口都用其中分数最高的作为代表。为了减少背景带来的影响，相同长度的窗口都使用top-K聚合函数。不同尺寸的窗口得到的各个分数投选出最终的结果。
* 作者举了个计算过程的例子，这里不再详述了。层这个方法为Multi-scale Temporal Window Integration（M-TWI）

***

## Experiments

### Datasets

##### Trimmed Video Datasets

* HMDB51：51个动作类，6766个clips，使用original评测方法。
* UCF101：101个动作类，13320个clips，使用THUMOS13 challenge评测。

##### Untrimmed Video Datasets

* THUMOS14: 101个动作类，1575个videos。分为train、valid、test和background四个部分。
* ActivityNet： 100个动作类，4819个videos for training，2383 videos for validation，2480 videos for testing。

模型使用mAP评测。

### Implementation Details

* mini-batch SGD
* batch size：128
* momentum：0.9
* ImageNet的pretrain模型。
* lr很小。
  * 在UCF101为0.001且枚1500次迭代缩减到十分之一。整体在3500次迭代后收敛。
  * 在temporal网络，初始化lr为0.005，12000和18000次迭代缩减到十分之一。最大迭代次数为20000.
* 八张TITANX八个小时。。。
* OpenCV的TVL1光流算法。

### Effectiveness of The Proposed Practices

##### Different learning strategy

* 相比于双流法，本项目使用了cross modality pre-training和有dropout的partia BN。作者对比了四种情况：

  1. 从零开始

  2. 只有pre-train spatial stream

  3. 有cross modality pre-training

  4. cross modality pre-training和有dropout的partia BN。

##### Different input modalities

* 作者提出了两种新的modalities：RGB difference和warped optical flow fields。

### Empirical Studies of Temporal Segment Networks

##### Evaluation on segment number

* K为对视频划分的segments数量，当K=1时模型退化为双流法。
* 作者测试了K=1～9，发现K越大效果越好。一般情况把K设置为7。

##### Evaluation on aggregation function

* 在较简单的数据（UCF101）上，average最好、weight average和attention weighting差不多。
* 在复杂的数据（ActivityNet）上，top-K和attention weighting表现更好。

##### Comparison of CNN architectures

* 对比了四种CNN：GoogleNet、VGG16、ResNet-152、BN-Inception。
* BN > ResNet > VGG16 > GoogleNet > VGG-M

### Comparison with The State of The Art

##### Trimmed Video Datasets

* HMDB51和UCF101
* 对打了
  * 传统方法：improved dense trajectories（iDTs）、MoFAP
  * 深度学习方法：C3D、TDD、FSTCN、LTC、KVMF

##### Untrimmed Video Datasets

* THUMOS14和ActivityNet v1.2
* 也打了一堆方法。。。









  

  


