# Multi-Context Attention for Human Pose Estimation

### CVPR 2017

## Abstract

* 将具有多内容信息注意力机制（multi-context attention mechanism）的 CNN 应用到人体姿势估计 end-to-end 框架中。

* 采用堆积沙漏网络（stacked hourglass network）生成多种分辨率特征的注意力图 。 
* 应用条件随机场（CRF）对注意力图中相邻区域之间的相关性进行建模。 作者考虑了整体注意模型和身体部位注意模型。 因此，作者的模型有能力专注于从局部显着区域到全局语义一致空间的不同粒度。 
* 设计了新颖的Hourglass Residual Unit（HRU），以增大网络的感受野，可以在 HRU 中学习并组合各种尺度的特征。 

## 1. Introduction

人体姿态检测面临肢体关联性、自我遮挡、服装影响、透视、背景干扰等问题的挑战。

针对这些问题，作者发现：

* ConvNets 可能由于背景干扰和自我遮挡产生错误
* 视觉注意力为识别人体关节的空间关系提供了思路
* 关节点注意力图可以解决重复计算问题（double counting problem），提高关节点识别精度。

已有文献证明，组合上下文信息对于 CV 任务有很重要的作用。

> 上下文信息是指：不同大小的图像区域中的特征信息。直观感受一下，较大的上下文区域能捕获全局空间信息，较小的上下文区域集中于局部特征外观。

之前常采用手动设计上下文信息表示的方法（如 multiple bounding boxes、multiple image crops 等），**作者将注意力机制用于生成上下文信息表示**。注意力机制不同于采用一系列矩形边界框手动定义关注区域，而是只依赖于图像特征，关注形状可变的区域。

<img src="https://i.loli.net/2020/09/05/wOkdqNCEutrSovD.png" alt="image-20200905202409932" />

上图就是注意力模型输出的注意力图，(b)、(c) 相比于 (a) 能够识别出被遮挡的腿，是因为注意力机制增加了上下文表示的信息量。**一般注意力模型才采用 Softmax 作为归一化的方法，作者设计了基于 CRF 的新型注意力模型。**该模型能更好地处理相邻区域的空间关联性。

作者采用 Stacked Hourglass Network 建立 Multi-Context Attention model。

***Stacked Hourglass Network***： 每个 Hourglass stack 处理特征图生成低分辨率的特征，低分辨率的特征经过 upsample 之后与原来的高分辨率特征结合。重复 Hourglass stack 逐渐捕获更全局化的特征表示。

 ***holistic attention models***：

1. 每个 Hourglass stack 中，对不同分辨率的特征生成多分辨率注意图（multi-resolutions attention maps）
2. 根据多个 hourglass stack，生成包含多层次语义信息的注意图，即多语义注意图（multi-semantic attention maps）。 

由于这些注意力图捕获了整个人体的空间信息，因此它们被称为*整体注意力模型*，整体注意力模型对遮挡和背景干扰有鲁棒性。

***part attention models***：整体注意力对关节点缺乏精确描述，因此扩展整体注意力到身体的各关节点。*部分注意力模型*

作者提出了一种 Hourglass Residual Units (HRU) 来替代 Residual Units，有更强的多尺度特征的表示能力，能更快地扩展感受野，与 stacked hourglass 模块一起构建 **nested hourglass networks**

文章贡献有三：

* 将注意力机制引入人体姿态估计，并改用 CRF
* 采用多上下文注意力机制使得模型有更好的鲁棒性和精度，包含三种 attention：
  * multi-resolution attention within each hourglass
  * multi-semantics attention across several stacks of hourglass
  * hierarchical visual attention scheme （分层视觉注意力机制，放大局部）

* 提出 HRU （hourglass residual units），与常规 Hourglass Units 一起构建嵌套Hourglass网络。

## 2. Related Work

* Human Pose Estimation
* Multiple Contextual Information
* Visual Attention Mechanism

## 3. Framework

![image-20200905230614850](https://i.loli.net/2020/09/05/9a4APizUuZly8Nh.png)

### 3.1 Baseline Network

作者采用 8-stack hourglass network 作为基线网络，它允许在每个 stack 末尾进行中间监督，重复地执行 bottom-up，top-down 跨尺度推断，实验中，输入图片尺寸是 $256 \times 256$ ，输出的 heatmaps 尺寸是 $K \times 64 \times 64$， 其中 $K$ 是身体部位的数目。作者采用 Mean Squared Error 作为损失函数。 

### 3.2 Nested Hourglass Networks

作者用 HRU 替换了残差单元，得到 nested hourglass network.

### 3.3 Multi-Resolution Attention

在每个 hourglass stack 中，多分辨率注意力图 $\Phi_r$ 根据不同比例的特征生成。其中 $r$ 是特征大小。

单个 hourglass stack 中的 multi-resolution attention。从不同分辨率特征图中生成多分辨率注意力图，这些注意图整合成单个注意力图，再进一步精炼得到 $h_1^{att}$。如下图：

![image-20200906000058155](https://i.loli.net/2020/09/06/mS8Q6wzNxfuD1Iv.png)

### 3.3 Multi-Semantics Attention

不同的 stack 有不同的语义：较低的 stack 侧重于局部特征，而较高的 stack 表示全局特征。

### 3.4 Hierarchical Attention Mechanism

分层注意力机制，在较低的 stack 中（stack1 至 stack4），作者使用两个整体注意力图 $h_1^{att}$ 和 $h_2^{att}$ 来编码整个人体的空间特征。 在较高的 stack 中（stack5 到 stack8 ），作者设计了一个 coarse-to-fine 的分层注意力机制，缩放局部节点。

## 4. Nested Hourglass Networks

### 4.1 Hourglass Residual Units

**回顾残差单元（Residual Units）**

Residual Units 可以用下式表示：
$$
\mathbf{x}_{n+1}=h\left(\mathbf{x}_{n}\right)+\mathcal{F}\left(\mathbf{x}_{n}, \mathbf{W}_{n}^{\mathcal{F}}\right)
$$

*  $\mathbf{x}_{n}$ 和 $\mathbf{x}_{n+1}$ 是第 n 个单元的输入和输出
* $\mathcal{F}$ 是 stacked convolution, batch normalization, and ReLU
*  $h\left(x_{n}\right)=x_{n}$ 是恒等映射（ identity mapping）

**沙漏残差单元（HRUs）**

实验证明上下文范围越大，定位身体部位越准确。作者给残差单元添加 micro hourglass branch 得到 HRU。在 stacked hourglass network 中使用 hourglass Residual units ，从而实现 **nested hourglass network** （宏观和微观都使用了 hourglass 结构）如下图：

![image-20200905232740713](https://i.loli.net/2020/09/05/o9eGw6v2RuXT7qd.png)

* A ：恒等变换分支
* B ：残差分支
* C ：HRU 分支

HRU 的形式化描述如下：
$$
\mathbf{x}_{n+1}=\mathbf{x}_{n}+\mathcal{F}\left(\mathbf{x}_{n}, \mathbf{W}_{n}^{\mathcal{F}}\right)+\mathcal{P}\left(\mathbf{x}_{n}, \mathbf{W}_{n}^{\mathcal{P}}\right)
$$

每个 HRU 都包含三个分支，

* 分支 A：式子中的  $\mathrm{x}_{n}$ 是 identity mapping（恒等映射），HRU 因此具有 ResNet 避免梯度消失的属性
* 分支 B：式子中的 $\mathcal{F}\left(\mathbf{x}_{n}, \mathbf{W}_{n}^{\mathcal{F}}\right)$ 是残差模块，同“回顾残差模块部分中的式子”
* 分支 C：式子中的  $\mathcal{P}\left(\mathbf{x}_{n}, \mathbf{W}_{n}^{\mathcal{P}}\right)$ 是作者的新设计，是 $2 \times 2$ max-pooling, 两个 $3 \times 3$ convolutions （带ReLU）, 和一个 upsampling 的网络层堆叠 

### 4.2 Analysis of Receptive Field of HRU

* 分支A 中的 identity mapping 感受野大小为$1 \times 1$。

* 分支B 中的residual block是卷积的堆叠 $\left(\mathrm{Conv}_{1 \times 1}+\mathrm{Conv}_{3 \times 3}+\mathrm{Conv}_{1 \times 1}\right) .$  对应 HRU 输入的 $3 \times 3$ 的区域. 

* 分支 C 的结构是（ $\mathrm{Pool}_{2 \times 2}+\mathrm{Conv}_{3 \times 3}+\mathrm{Conv}_{3 \times 3}+Deconv_{2 \times 2}$）分辨率是 A、B 的一半，感受野约为 B 的三倍。

具有不同接受域和分辨率的这三个分支被加在一起作为HRU的输出。 因此，HRU单元通过包含分支（C）来增加接收字段的大小，同时通过使用分支（A）和（B）来保留高分辨率信息。

## 5. Attention Mechanism

### 5.1 Conventional Attention

用 $f$ 表示卷积特征，第一步要生成汇总的卷积图：
$$
s = g(W^{a} * f + b)
$$

* $W^a * f$ 表示卷积操作
* $g()$ 表示非线性激活函数
* $s \in \mathbb R ^{H \times W} $ 整合f 的所有通道的信息。 

$s(l)$ 表示 feature map 上 $l=(x, y)$ ,位置处的特征 s 。将 Softmax 应用在 $ s(l)$ 上，得到下式：
$$
\Phi(l)=\frac{e^{\mathbf{s}(l)}}{\sum_{l^{\prime} \in \mathbb{L}} e^{\mathbf{s}\left(l^{\prime}\right)}}
$$
* $\mathbb{L}=\{(x, y) \mid x=1, \ldots, W, y=1, \ldots, H\}$ . 
* $\Phi$ 是attention map, 其中 $\sum_{l \in \mathbb{L}} \Phi(l)=1$. 

将 attention map 作用到 特征 $f$ 上：

$$
\mathbf{h}^{\text {att }}=\Phi \star \mathbf{f}, \quad \text { where } \mathbf{h}^{\mathrm{att}}(c)=\mathbf{f}(c) \circ \Phi
$$
* $c$ 是特征通道的索引
* $\star$  表示channel-wise Hadamard 矩阵乘法运算. 
* $\mathrm{h}^{\mathrm{att}}$ 是 refined feature map, $h^{att}$ 由 attention map $\Phi$ 分配权重, 与特征图 $f$ 的形状相同.

### 5.2  Multi-Context Attention Model

作者对注意力模型进行了以下几个修改。 

1. 用 CRF 替换 Softmax 。 
2. 根据不同分辨率的特征生成关注图，以使模型更加稳健。 
3. 为每个 hourglass stack 生成注意图来获得多语义 attention 。
4. 使用分层的从粗到细（即从全身到部位）注意力机制，以放大身体局部关节区域，进行更精确的定位。

**Spatial CRF Model**

使用 CRF 对空间相关性进行建模，使用 mean-field approximation 的方法递归学习空间相关核。

attention map 被建模成二分类问题，用 $y_{l}=\{0,1\}$ 作为 第 i 个位置的 attention 标签。在 CRF 模型中， 标签分配$y = \left\{y_{l} \mid l \in \mathbb{L}\right\}$ 如下表示：
$$
E(\mathbf{z})=\sum_{l} y_{l} \psi_{u}(l)+\sum_{l, k} y_{l} w_{l, k} y_{k}, \quad
$$

* $\psi\left(y_{l}\right)=g(\mathbf{h}, l)$ 
* 给定图像 I 标签分配的概率 y 是 $P(\mathrm{y}\mid\mathrm{I})=\frac{1}{Z} \exp (-E(\mathbf{y} \mid \mathbf{I}))$

$y_l = 1$ 的概率如下：
$$
\Phi\left(y_{l}=1\right)_{t}=\sigma\left(\psi_{u}(l)+\sum_{k} w_{l, k} \Phi\left(y_{k}=1\right)_{t-1}\right), \quad \begin{array}{l}
\end{array}
$$

* $\sigma(a)=1 /(1+\exp (-a))$ 是 sigmoid 函数
* $\psi_{u}(l)$ 由特征 h 卷积得到
* $\sum_{k} w_{l, k} \Psi\left(y_{j}=1\right)$  是通过将 t-1 阶段的 attention map $\Phi_{l-1}$ 做卷积得到的，初始时，$\Phi\left(y_{i}=1\right)_{1}=\sigma\left(\psi_{u}(i)\right)$

综上所述 t 阶段的 attention map 表示如下：
$$
\Phi_{t}=\mathcal{M}\left(\mathrm{s}, \mathbf{W}^{k}\right)=\left\{\begin{array}{ll}
\sigma\left(\mathbf{W}^{k} * \mathrm{s}\right) & t=0 \\
\sigma\left(\mathbf{W}^{k} * \Phi_{t-1}\right) & t=1,2,3
\end{array}\right.
$$

* $\mathcal{M}$ 表示 mean field approximation 中的权值共享卷积序列。
* $W^{k}$ 表示空间关联核，$W^{k}$ 在不同 time steps 之间共享，作者采用三步递归卷积。

**Multi-Resolution Attention**

![image-20200905234801124](https://i.loli.net/2020/09/05/1Ktqsg5W2k9dYlR.png)

如上图所示，upsample 过程中会生成大小 r 不同的特征 $f_r, \; r \in \{8,16,32,64\}$ ；$s_r$ 用于生成 attention maps $\Phi_r$ ，$\Phi_r$ 被上采样到大小为 $64 \times 64$ 记做 $\Phi_{\{r \rightarrow 64 \}}$。这些 attention maps 对应不同的分辨率，例如 $\Phi_{\{8 \rightarrow 64 \}}$ 具有较小的分辨率，着重于身体的整体信息；$\Phi_{64}$ 具有较高的分辨率，着重于身体局部信息。

所有的上采样 attention maps 加到一起，作用特征 f ：
$$
\mathbf{h}_{1}^{\text {att }}=\mathbf{f} \star\left(\sum_{r=8,16,32,64} \Phi_{\{r \rightarrow 64\}}\right)
$$

* 特征 f 是 stacked hourglass 最后一层的输出
*  $\star $ 运算符，如下定义 $h^{att}=\Phi \star f$，其中 $h^{att}(c)=f(c) \circ \Phi$

一般 attention maps 常常直接用于 refine 生成它的 feature ，但是这样会导致很多的接近 0 的值，不利于反向传播计算。因此作者采用 **由不同分辨率 feature 生成 attention maps 将其作用于最后的 feature**

从 $h_{1}^{att}$ 生成了 attention map $\Phi^{\prime}$ 和它对应的 refine feature $h_2^{att}$ :
$$
\mathbf{h}_{2}^{\text {alt }}=\mathbf{h}_{1}^{\text {att }} \star \Phi^{\prime}=\mathbf{h}_{1}^{\text {att }} \star \mathcal{M}\left(\mathbf{h}_{1}^{\text {att }}, \mathbf{w}\right)
$$
**Multi-Semantics Attention**

在 hourglass stack 上重复上述过程，以生成具有多种语义含义的 attention maps。 下图显示了 $\Phi^{\prime} $ 从 stack1 到stack8  的样本。较浅的 hourglass stack 的 attention map 捕获了更多的局部信息。 对于更深的 hourglass stack，将捕获整个人的全局信息，这对于物体遮挡有更好的鲁棒性。

![image-20200907002308811](https://i.loli.net/2020/09/07/GF5tYviT3xJ9KnI.png)

**Hierarchical Holistic-Part Attention**

在 第 4 到 8 个 hourglass stack 中，作者使用 refine feature $h_1^{att}$ 生成身体部位 attention maps 如下： 
$$
\begin{aligned}
\mathrm{s}_{p} &=g\left(\mathbf{W}_{p}^{a} * \mathbf{h}_{1}^{\text {att}}+\mathbf{b}\right) \\
\Phi_{p} &=\mathcal{M}\left(\mathbf{s}_{p}, \mathbf{W}_{p}^{k}\right)
\end{aligned}
$$
* $p \in\{1, \cdots, P\}$
* $W_{p}^{a}$  表示用于获取人体部位 p 的 summarization map $\mathrm{s}_{p}$ 的参数
*  $\mathbf{W}_{p}^{k}$  表示部位 p 的空间建模
* 部位 attention map $\Phi_{p}$ 和 refined feature map $\mathbf{h}_{1}^{\text {att }}$ 结合生成部位 $p$ 的 refined feature map 如下:

$$
\mathbf{h}_{p}^{att }=\mathbf{h}_{1}^{att} * \Phi_{p}
$$



第 p 个人体关节的 heatmap 基于 refined features $h^{att}_p$ 得到，如下所示：
$$
\hat{\mathbf{y}}_{p}=\mathbf{w}_{p}^{\mathrm{cls}} * \mathbf{h}_{p}^{\mathrm{att}}
$$

* $\hat y_p$ 是身体部位 p 的 heatmap
* $w^{cls}_p$ 是分类器。  这样保证attention map $\Phi_p$ 是身体部位 p 特有的。

![image-20200907131237364](https://i.loli.net/2020/09/07/hlnUI6o8WrKTOuD.png)

## 6. Training the model

每个 hourglass stack 生成一个预测的 heatmap，计算每个 stack 的 MSE loss ，公式如下：
$$
L= \sum_{p=1}^P \sum_{l \in \mathbb L}||\hat y_p(l) - y_p(l)||_2^2
$$

* p 表示身体部位，$l$ 表示第 $l$ 个位置
* $\hat y_p$ 表示对部位 p 的 heatmap
* $y_p$ 是 ground-truth heatmap (2-D Gaussian 生成)

## 7. Experiments

**Dataset** ：MPII + LSP

**Data Augmentation**：在训练过程中，作者对图像做分割、缩放、旋转、颜色抖动实现数据增强。

**Experiment Settings**：训练模型时，初始学习率采用 $ 2.5 \times 10^{-4}$。 通过 RMSprop 算法优化参数。 在 MPII 数据集训练 130 epochs 在 LSP 数据集训练 60 个 epochs 。

### 7.1 Results

作者在 LSP上用Percentage Correct Keypoints (PCK)  评估模型，在 MPII 上采用 PCKh 评估。