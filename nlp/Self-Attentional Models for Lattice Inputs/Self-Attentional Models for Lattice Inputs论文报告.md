# Self-Attentional Models for Lattice Inputs

## Abstract

* Lattice在复杂的并且具有歧义的下游任务中具有非常好的效果。就比如在中文NER任务中表现很出色的模型Lattices-LSTM。但是作者指出，之前的大多数Lattice会结合在RNN中，而RNN的缺点就是计算速度很慢。所以本文尝试使用Self-Attention代替RNN来提升计算速度和精度。
* 本文的贡献包括：
  * 使用 reachability masks来良好的结合lattices和attention。
  * 新的位置编码方法。
* 最终模型被应用在一个演讲翻译的任务中，并且无论在训练还是预测过程中速度都远超其他baseline。

## Introduction

* 在很多NLP任务中基于图的模型都被证实非常适合处理高度结构化的信息，Lattices就是其中之一。Lattice因为能够捕获大量的特征信息，在很多可选信息非常丰富的任务中表现出色，比如speech recognition、word segmentation、word class甚至video descriptions。
* 受到早期将RNN扩展到树形结构研究的影响，近年来Lattice和RNN结合的非常好。但是这些网络都有一个共性的缺点，就是RNN树形结构需要高昂的计算代价。而其代替品GCN计算速度更快，但是其只能考虑到局部信息，所以在多数NLP任务仍无法摆脱RNN。
* Attention机制已经发展到能够代替RNN的阶段，注意力机制主要通过考量类似相似度信息的方法调整不同状态下的注意力集中位置，同时由于使用了位置编码，能够非常高效的并行计算。
* 作者的目标是在保留lattice-RNN方法灵活性能的同时尽可能减少计算开销。主要贡献有两点：
  * 通过使用reachability mask来模仿之前RNN的pairwise condition，从而很好的融合了全局attention。
  * 提出了lattice的位置编码。
* 最终的实验表明，Lattice-Attention模型在两个翻译任务中具有比LSTM-base和LatticeLSTM模型更出色的效果，并且有更快的计算速度。

## Background

### Masked Self-Attention

* 注意力机制可以简述为对三个不同部分的维护：query, key, value。不同于朴素attention，self-attention使用序列中的每一个成员作为query，去查询与其他成员的key的相关程度，从而控制value。公式为：

  > $$\begin{aligned} e_{i j} &=f\left(q\left(\mathbf{x}_{i}\right), k\left(\mathbf{x}_{j}\right)\right)+m_{i j} \quad(\forall 1 \leq j \leq l) \\ \boldsymbol{\alpha}_{i} &=\operatorname{softmax}\left(\mathbf{e}_{i}\right) \\ \mathbf{y}_{i} &=\sum_{j=1}^{l} \alpha_{i j} v\left(\mathbf{x}_{j}\right) \end{aligned}$$
  
  式中的m参数可以达到mask的效果，这一点将在后文详细讨论。

### Lattices

* 首先构造了一个有向无环图，添加了源点和汇点，从而将每一个序列都表示为一条从源点到汇点的路径，每一个节点都是一个word token。使用G=(V, E)来表示图，$\mathcal{R}_{\mathrm{G}}^{+}(k)$表示k可到达的点，$\mathcal{N}_{\mathrm{G}}^{+}(k)$表示与k相邻的点。$j \succ i$表示j对于i是可到达的。
* 图上的转移概率：$p_{G}(j \succ i \mid i)$表示从i转移到j的概率。从而有了整个lattice的转移概率：$p_{k, j}^{\mathrm{trans}}:=p_{G}(k \succ j \mid j)$ for $j \in \mathcal{N}_{\mathrm{G}}^{+}(k)$

## Baseline Model

### Lattice-Biased Attentional Decoder

* 模型整体采用Encoder-Decoder架构。其中使用了lattice-biased变体的attention。通过编码器i和解码器j的marginal lattice分数（计算方法在下文）来调整$\alpha_{i j}^{\mathrm{cross}}$, 具体公式为：

	> $$\alpha_{i j}^{\mathrm{cross}} \propto \exp (\operatorname{score}(\bullet)+\log p(j \succ \mathrm{S} \mid \mathrm{S}))$$

  其中$\operatorname{score}(\bullet)$为attention的socre。
  
* 虽然作者表示应该使用类似self-attention的decoder。但是总体的Decoder使用LSTM，和一些处理：input feeding、variational dropout in the decoder LSTM和label smoothing。
  
### Mutil-Head Transformer Layers

* 整体设计参照了经典的Transformer模型，由于Transformer非常经典，所以我这里就不详细列举公式了。作者此处着重提及了Transformer模型的多头attention能够分开添加mask。和其中使用的位置编码方法（其实我也认为简单的三角函数编码并不能很好的表达应有的位置信息），而本文使用了pre train的position embedding。

## Self-Attentional Lattice Encoders

从topological order理解和实现lattice模型是很简单直接的，但是会忽视数据的结构信息并且相关的query和key不能够很好的在lattice中同时表现。作者验证了这个方法的缺点，同时提出了新的mask策略。

### Lattice Reachability Masks

#### Binary Masks

* 提出了和token base lattice结构结果相似的mask策略：

  > $$\begin{array}{ll}\vec{m}_{i j}^{\mathrm{bin}}=\left\{\begin{array}{ll}0 & \text { if } i \in \mathcal{R}^{-}(j) \vee i=j \\ -\infty & \text { else }\end{array}\right. \\ \overleftarrow{m}_{i j}^{\mathrm{bin}}=\left\{\begin{array}{ll}0 & \text { if } i \in \mathcal{R}^{+}(j) \vee i=j \\ -\infty & \text { else }\end{array}\right.\end{array}$$

  简单来说就是在前后向传播时mask掉当前节点的前驱节点。

#### Probabilistic Masks

* Binary Mask能够抽取图的结构信息，但是缺少了概率信息。之前的一些工作得到抽取lattice scores是非常重要的，尤其speech recognition这些输入中噪声很大的任务中。Binary Mask对所有节点的效果是平等的，也就导致了不同置信度影响因素产生的效果差异并没有很大。

* 所以提出了另一种mask策略：

  > $$\begin{array}{l}\vec{m}_{i j}^{\mathrm{prob}}=\left\{\begin{array}{ll}\log p_{G}(j \succ i \mid i) & \text { if } i \neq j \\ 0 & \text { if } i=j\end{array}\right. \\ \overleftarrow{m}_{i j}^{\mathrm{prob}}=\left\{\begin{array}{ll}\log p_{G^{\top}}(j \succ i \mid i) & \text { if } i \neq j \\ 0 & \text { if } i=j\end{array}\right.\end{array}$$

  mask参数使用对数空间的转移概率来表示，对于那些无法转移的节点，也就是在Binary Mask中被置为负无穷的节点转移概率为0，所对应的对数空间同样被设置为负无穷。整个图的前向和后向mask都能够在$O(n^3)$时间内完成。

#### Directional and Non-Directional Masks

* 在probabilistic mask中得到了前向和后向两种概率，就有了两种mask策略：
  * Non-Directional Mask：不考虑方向，仅选取二者中较大的。
  * Directional Mask：考虑方向，在Multi-Head中一般采用前向Mask，一般采用后向Mask。
* 当输入仅为一段序列时，无向Mask退化为没有Mask的self-attention。有向Mask退化为之前提出的一种序列Mask。

### Lattice Positional Encoding

位置编码是Attention模型中尤为重要的部分，作者为编码策略提出了几点要求：

> * 编码必须是整数，所以positional embedding能够应用在这里。
> * 每一条可能的lattice路径上编码都必须是严格单调递增的。
> * 为了简洁，要避免所有不是必须的编码跳跃。

* 所以采用从源点开始的最长路径编码(ldist)：

  > $$\mathbf{x}_{i}^{\prime}=\operatorname{dropout}\left(\mathbf{x}_{i}+\text { embed }[\text { ldist }(\mathbf{S} \rightarrow i)]\right)$$

* 该策略能够满足上述的三点要求，同时能够在$O(n^2)$时间内解决。

## Experiments

模型选择在speech translation任务上评测。所以decoder输出更换为翻译的目标语言。

### 模型和数据：

* 数据集：Fisher–Callhome Spanish–English Speech Translation corpus
* tokenize ， lowercase data, remove punctuation, and replace singletons with a special unk token
* Hidden dimensions：512
* LSTMbased decoder：单层、Dropout 0.5
* attention之后三层FF：2048、Dropout 0.1
* LSTM-based encoders： 两层。

### 训练：

* pertrain and fine tuning参照另一篇文章。
* Adam，warm-up和decay参照另一篇文章。
* lr：0.0001
* batch size：1024
* 评价方法：BLEU

### Main Results

* 本模型比self-attention模型提升了1.31-1.74个点，比LatticeLSTM提升了0.64-0.73个点。
* 在拥有更多层数的同时具有更快的运行速度。训练速度提升近一倍，测试速度略有提升。

***

## 总结

* 图网络在深度学习中具有非常重要的意义，而且在各个领域都不断的有突破和提升。无论是经典的如CRF的概率图模型，还是图卷积或者是知识蒸馏。图网络都能表现出非常全面的结构信息，这在知识工程、NLP甚至CV都有很多应用。这是非常值得学习和研究的。
* Attention机制近年来发展非常迅速，无论是Transformer还是BERT，或者是图像Attention、图Attention甚至视频的Attention都在对应的领域推动了非常重要的一步。从某种角度来讲，注意力机制和LSTM具有相似的特性，但同时具有根本的不同。就我个人的理解，注意力更偏向于表示个体对整体的影响，无论是Attention还是Self-Attention，都是站在不同个体上考虑其对全局其他个体的影响。而LSTM则更多是从整体上考虑问题，是在维护整体的同时，对每一个个体的信息作出修正。这两种模型无疑都是非常出色的模型，而其背后的设计动机同样值得学习和借鉴。
* Lattice是NLP模型中非常重要的一个，尤其在中文任务中，不同于英文先天完成分词，中文的很多任务需要在pipline前面先行经过分词。就我个人经历而言，无论是序列标注、序列分类或者是序列生成的任务中，尤其是小参数和小网络里面先分词绝对是提升数据的有效手段。但是使用分词jiu必定要接受分词带来的累积误差，简单的步骤叠加是一个简单的解法，但绝对不是最适合的解法。其实在我第一次看到LatticeLSTM之前就考虑过通过某种门控方法或者筛选手段来控制指数级别的组合数量，而LatticeLSTM正好是解决了这些所有的问题。这也是我这次选择做Lattice-Attention报告的原因。
* 在阅读Transformer论文和代码的时候我就认为原论文使用的三角函数位置编码略有不足，虽然从反馈相对位置关系的角度来看这个方法完成得非常出色，但是总觉得周期函数来的会不那么直观。本文中提及的新的编码方法可以算是一个启发。而至于更多的编码方式和更近一步的对比还需要更加努力的学习和研究。
* 看的paper还太少，该学的还太多，我就是个啥也不会的菜鸡。。。



  





