## 概念

Transformer模型是一种基于注意力机制的神经网络模型，核心思想是通过自注意力机制（Self-Attention）来捕捉输入序列中各个位置之间的依赖关系，避免了传统循环神经网络（RNN）在处理长距离依赖时的困难。

Transformer总体上可以分为三个阵营

1. BERT。组员都是BERT类似的结构，是一类自编码模型。
2. GPT。组员都是类似GPT的结构，是一类自回归模型。
3. BART/T5。组员结构都是encoder-decoder模型。

<img src="https://raw.githubusercontent.com/AnJian2020/study_recorder/main/images/202308151051193.png" alt="image-20230815105145107" style="zoom:50%;" />

## 组成部分

Transformer主要由两部分组成：编码器（Encoder）和解码器（Decoder）。

- 编码器将输入序列中的每个词转换为其对应的词向量表示，并且引入自注意力机制来对整个输入序列进行建模，从而将上下文信息进行编码。自注意力机制通过计算输入序列中不同位置之间的相关性得分，来决定每个位置对其他位置的关注度，进而加权地聚合上下文信息。

- 解码器在编码器的基础上，进一步引入另一个注意力机制，称为“编码-解码注意力”（Encoder-Decoder Attention）。它使解码器能够利用编码器输出的信息来为生成下一个词提供上下文参考，以便更好地处理输出序列的依赖关系。

<img src="https://raw.githubusercontent.com/AnJian2020/study_recorder/main/images/202308151053256.png" alt="image-20230815105348124" style="zoom:50%;" />

编码组件部分由一堆编码器 （encoder）构成。 解码组件部分也是由相同数量（与编码器对应）的解码器（decoder）组成的。

<img src="https://raw.githubusercontent.com/AnJian2020/study_recorder/main/images/202308151055850.png" alt="image-20230815105552750" style="zoom:50%;" />

> 解码器、编码器输入输出的数据维度一致。

从编码器输入的句子首先会经过一个自注意力（self-attention） 层，编码器在对每个单词编码时关注输入句子的其他单词。

自注意力层的输出会传递到前馈 （feed-forward）神经网络中。 每个位置的单词对应的前馈神经网络都完全一样（一层窗口为一个单词的一维卷积神经网络）。

<img src="https://raw.githubusercontent.com/AnJian2020/study_recorder/main/images/202308151109714.png" alt="image-20230815110921611" style="zoom:50%;" />

所有的编码器在结构上都是相同的，但它们没有共享参数。每个编码器都可以分解成两个子层。

> Transformer的核心特性：
>
> - 输入序列中每个位置的单词都有自己独特的路径流入编码器。 在自注意力层中，这些路径之间存在依赖关系。
> - 前馈（feed-forward）层没有这些依赖关系。
> - 在前馈（feed-forward）层时可以并行执行各种路径。

<img src="https://raw.githubusercontent.com/AnJian2020/study_recorder/main/images/202308151111763.png" alt="image-20230815111114697" style="zoom:50%;" />

模型需要对输入的数据进行一个embedding操作 

enmbedding结束之后，输入到encoder层，self-attention处理完数据后把数据送给前馈神经网络。

前馈神经网络的计算可以并行，得到的输出会输入到下一个encode。

解码器中也有编码器的自注意力（self-attention）层和前馈（feed-forward）层。

这两个层之间还有一个注意力层，用来关注输入句子的相关部分（和seq2seq模型的注 意力作用相似）

<img src="https://raw.githubusercontent.com/AnJian2020/study_recorder/main/images/202308151434163.png" alt="image-20230815143448029" style="zoom:50%;" />

### 将张量引入图景

- 首先将每个输入单词通过词嵌入算法转换为词向量。

![image-20230815143719741](https://raw.githubusercontent.com/AnJian2020/study_recorder/main/images/202308151437805.png)

- 每个单词都被嵌入为512维的向量，用简单的方框来表示这些向量。
- 词嵌入过程只发生在最底层的编码器中。所有的编码器都有一个相同的特点，即它们接收一个向量列表，列表中的每个向量大小为512维。
- 在底层（最开始）编码器中它就是词向量，但是在其他编码器中，它就是下一层编码器的输出（也是一个向量列表）。向量列表大小是我们可以设置的超参数——一般是训练集中最长句子的长度。

### 使用位置编码表示序列顺序

位置嵌入（ Positional Encoding），位置嵌入的维度为 [max_sequence_length, embedding_dimension], 位置嵌入的维度与词向量的维度是相同的，都是 embedding_dimension。

max_sequence_length 属于超参数，限定每个句子最长由多少个词构成。

> 一般以字为单位训练Transformer 模型。首先初始化字编码的大小为[vocab_size,embedding_dimension]，vocab_size 为字库中所有字的数量，embedding_dimension 为字向量的维度。

添加位置：encoder层和decoder层的输入，维度和embedding的维度一样，这个向量能决定当前词的位置，或序列中不同单词之间的距离。

位置向量的计算方法：

<img src="https://raw.githubusercontent.com/AnJian2020/study_recorder/main/images/202308151440361.png" alt="image-20230815144051288" style="zoom:50%;" />

**优点**：能够扩展到未知的序列长度

将位置向量添加到词嵌入中使得它们在接下来的运算中，能够更好地表达的词与词之间的距离。

<img src="https://raw.githubusercontent.com/AnJian2020/study_recorder/main/images/202308151442605.png" alt="image-20230815144239508" style="zoom:50%;" />

位置编码向量可以让模型理解单词的顺序，这些向量遵顼特定的模式。

> 如果假设词嵌入的维数为4，则实际的位置编码如下
>
> <img src="https://raw.githubusercontent.com/AnJian2020/study_recorder/main/images/202308151444306.png" alt="image-20230815144451216" style="zoom:50%;" />
>
> 在图中，每一行对应一个词向量的位置编码，所以第一行对应着输入序列的第一个词。每行包含 512个值，每个值介于1和-1之间。对它们进行颜色编码，图案如下：
>
> ![image-20230815144550867](https://raw.githubusercontent.com/AnJian2020/study_recorder/main/images/202308151445980.png)
>
> 上面有一组sin 和cos 公式，也就是 对应着 embedding_dimension 维度的一组奇数和偶数的序号的维度，例如 0,1 一组，2,3 一组，分别用上面的 sin 和cos 函数做处理，从而产生不同的周期性变化。
>
> 位置嵌入在 embedding_dimension 维度上随着维度序号增大，周期变化会越来越慢，最终产生一种包含位置信息的纹理。
>
> 上图从中间分裂成两半。这是 因为左半部分的值由一个函数(使用正弦)生成，而右半部分由另一个函数(使用余弦)生成。然后将它们拼在一起而得到每一个位置编码向量。

### 自注意力机制

着模型处理输入序列的每个单词，自注意力会关注整个输入序列的所有单词，帮助模型对本单词更好地进行编码，会将所有相关单词的理解融入到正在处理的单词中。

**计算自注意力**

**第一步**

从每个编码器的输入向量（每个单词的词向量）中生成三个向量——一个查询向量Query 、一个键向量Key和一个值向量Value。这三个向量是通过词嵌入与三个权重矩阵后相乘创建的。这个权重矩阵是随机初始化的，维度为（64，512）

> - 可以发现这些新向量在维度上比词嵌入向量更低。
> - 词嵌入和编码器的输入向量的维度是512。
> - 使多头注意力（multiheaded attention）的计算保持不变

与单词相关的查询向量如下：

$$
q_1=X_1*W^Q
$$

![image-20230815145819826](https://raw.githubusercontent.com/AnJian2020/study_recorder/main/images/202308151458916.png)

**第二步**

计算得分。

假设为第一个词“Thinking”计算自注意力向量，需要拿输入句子中的每个单词 对“Thinking”打分。这些分数决定了在编码单词“Thinking”的过程中有多重视句子的其它部分。这些分数是通过打分单词（所有输入句子的单词）的键向量与“Thinking”的查询向量相点积来计算的。如果处理位置最靠前的词的自注意力，第一个分数是q1和k1的点积，第二个分数是q1和k2的点积

![image-20230815150010252](https://raw.githubusercontent.com/AnJian2020/study_recorder/main/images/202308151500378.png)

**第三步和第四步**

将分数除以8(8是论文中键向量的维数64的平方根，这会让梯度更稳定。也可以使用其它值，8是默认值)，然后通过softmax传递结果。

> - softmax的作用是使所有单词的分数归一化，得到的分数都是正值且和为1。
> - 这个softmax分数决定了每个单词对编码当下位置（“Thinking”）的贡献。 在这个位置上的单词将获得最高的softmax分数，有时关注另一个与当前单词相关的单词也会有帮助。

![image-20230815150210744](https://raw.githubusercontent.com/AnJian2020/study_recorder/main/images/202308151502832.png)

**第五步**

将每个值向量乘以softmax分数。（希望关注语义上相关的单词，并弱化不相关的单词）

**第六步**

