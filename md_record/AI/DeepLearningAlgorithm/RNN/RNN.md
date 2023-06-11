# RNN

### 1. 简述RNN，LSTM，GRU的区别和联系

> 参考： https://blog.csdn.net/softee/article/details/54292102

> RNN(Recurrent Neural Networks,循环神经网络)不仅学习当前输入的信息，还要依赖之前的信息，如处理由重多词组成的序列。
>
> 但它并不能很好地处理长的序列。是因为会出现梯度消失和梯度爆炸现象，由此出现了LSTM。
>
> LSTM和RNN相同都是利用BPTT传播和随机梯度或者其他优化算法来拟合参数。
>
> 但是RNN在利用梯度下降算法链式求导时是连乘的形式,而LSTM是相加的形式，这也是最主要的区别。
>
> GRU与LSTM相比，少了一个gate,由此就少了一些矩阵乘法，GRU虽与LSTM的效果较近，但却因此在训练中节省了很多时间，在文本类处理中，相比用LSTM，导师更建议用GRU来训练。
>
> 最小GRU可看最近南大教授的篇论文是用最小GRU来进行计算的，比GRU又少了一个门，这就使计算时间更短了，计算维度降低，占用内存减少。

* RNN 

    > 在传统的神经网络中，输入是相互独立的，但是在RNN中则不是这样。一条语句可以被视为RNN的一个输入样本，句子中的字或者词之间是有关系的，后面字词的出现要依赖于前面的字词。RNN被称为并发的（recurrent），是因为它以同样的方式处理句子中的每个字词，并且对后面字词的计算依赖于前面的字词。
    >
    > 典型的RNN如下图所示：
    >
    > ![](imgs/deep_learning/RNN/RNN_structure.png)
    >
    > 图中左边是RNN的一个基本模型，右边是模型展开之后的样子。展开是为了与输入样本匹配。
    >
    > > * $ x_{t} $代表输入序列中的第t步元素，例如语句中的一个汉字。一般使用一个one-hot向量来表示，向量的长度是训练所用的汉字的总数（或称之为字典大小），而唯一为1的向量元素代表当前的汉字。
    > > * $ s_{t} $代表第t步的隐藏状态，其计算公式为$s_{t}=tanh(UX_{t} + WS_{t-1})$。也就是说，当前的隐藏状态由前一个状态和当前输入计算得到。考虑每一步隐藏状态的定义，可以把$s_{t}$视为一块内存，它保存了之前所有步骤的输入和隐藏状态信息。$s_{-1}$是初始状态，被设置为全0。
    > > * $o_{t}$是第t步的输出。可以把它看作是对第 t+1 步的输入的预测，计算公式为：$o_{t} = softmax(Vs_{t})$。可以通过比较$o_{t}$ 和$x_{t+1}$之间的误差来训练模型
    > > * U, V, W 是RNN的参数，并且在展开之后的每一步中依然保持不变。这就大大减少了RNN中参数的数量
    >
    > * 举例
    >
    >     > 假设我们要训练的中文样本中一共使用了3000个汉字，每个句子中最多包含50个字符，则RNN中每个参数的类型可以定义如下。
    >
    >     > * $x_{t} \in R^{3000}$ , 第t步的输入，是一个one-hot向量，代表3000个汉字中的某一个。
    >     > * $o_{t} \in R^{3000}$,第t步的输出，类型同$x_{t}$
    >     > * $s_{t} \in R^{50}$，第t步的隐藏状态，是一个包含50个元素的向量。RNN展开后每一步的隐藏状态是不同的。
    >     > * $U \in R^{50 * 3000}$，在展开后的每一步都是相同的。
    >     > * $V \in R^{3000 * 50}$，在展开后的每一步都是相同的。
    >     > * $W \in R^{50 * 50}$，在展开后的每一步都是相同的。
    >     >
    >     > 其中，$x_{t}$是输入，$U, V, W$是参数，$s_{t}$是有输入和参数计算所得到的隐藏状态，而$o_{t}$是输出。
    >
    > * 

* LSTM

    > LSTM是为了解决RNN中的反馈消失问题而被提出的模型，它也可以被视为RNN的一个变种。与RNN相比，增加了3个门（gate）：input门，forget门和output门，门的作用就是为了控制之前的隐藏状态、当前的输入等各种信息，确定哪些该丢弃，哪些该保留，如下图表示：
    >
    > ![](imgs/deep_learning/RNN/LSTM_stucture.png)
    >
    > LSTM的隐藏状态g的计算公式：$g = tanh(U_{g}X_{t} + W_{g}s_{t-1})$ 。但是这个隐藏状态的输出受各种门的限制。
    >
    > 内部存储用$c$来表示，它由前一步的内部存储和当前的隐藏状态计算得出，并且受到input门和forget门的控制。前者确定当前隐藏状态中需要保留的信息，后者确定前一步的内部存储中需要保留的信息:$c_{t}= c_{t-1}·f + g · i$。 LSTM的输出则使用$s_{t}$ 表示，并且受输出们的限制：$s_{t} = tanh(c_{t}) · o$。综上所述，第t步的LSTM种输出信息的计算公式如下：
    >
    > * $i = \sigma(U_{i}X_{t} + W_{i}S_{t - 1})$
    > * $f = \sigma(U_{f}X_{t} + W_{f}S_{t - 1})$
    > * $o = \sigma(U_{o}X_{t} + W_{o}S_{t - 1})$
    > * $g = tanh(U_{g}X_{t} + W_{g}S_{t - 1})$
    > * $c_{t}= c_{t-1}·f + g · i$
    > * $s_{t}= tanh(c_{t})·o$
    >
    > 公式中的变量$i,f,o,g,c_{t}$的数据类型与$s_{t}$一样，是一个向量。圆点表示向量之间逐个元素相乘而得到一个新向量。这些式子具有以下特点。
    >
    > * 三个门input、forget、output具有相同的形式，只是参数不同。它们各自的参数$U,W$都需要在对样本的训练过程中学习。
    > * 隐藏状态$g$的计算与RNN中的隐藏状态相同，但是不能直接使用，必须通过input门的约束，才能够作用到内部存储$c_{t}$之中。
    > * 当前的内部存储的计算，不仅依赖于当前的隐藏状态，也依赖于前一步的内部存储$c_{t-1}$，并且$c_{t-1}$受forget门的约束。
    > * 输出信息在$c_{t}$的基础上又施加了一层tanh函数，并且受到输出门的约束。
    > * 如果input门全为1，forget门全为0，output门全为1的话，则LSTM与RNN相似，只是多了一层tanh函数的作用。

    总之，门机制的存在，就使得LSTM能够显示地为序列中长距离的依赖建模，通过对门参数的学习，网络能够找到合适的内部存储行为。

* GRU

    > GRU具有与LSTM类似的结构，但是更为简化，如下图所示。
    >
    > ![](imgs/deep_learning/RNN/GRU_structure.png)
    >
    > GRU中状态与输出的计算包含以下步骤。
    >
    > * $z = \sigma(U_{z}x_{t} + W_{z}s_{t - 1})$
    > * $r = \sigma(U_{r}x_{t} + W_{r}s_{t - 1})$
    > * $h = {\rm tanh}(U_{h}x_{t} + W_{h}(s_{t - 1} · r))$
    > * $ s_{t} = (1 - z) · h + z · s_{t - 1}$

    * 与LSTM相比，GRU存在着下述特点。
        * 门数不同。GRU只有两个门reset门r和update门z。
        * 在GRU中，r和z共同控制了如何从之前的隐藏状态（$s_{t-1}$）计算获得新的隐藏状态（$s_{t}$），而取消了LSTM中的output门。
        * 如果reset门为1，而update门为0的话，则GRU完全退化为一个RNN。

### 2. 画出lstm的结构图，写出公式

> 参考问题 1

### 3. RNN的梯度消失问题？如何解决？

* 梯度消失与梯度爆炸的原因

    > RNN的梯度是多个激活函数偏导乘积的形式来计算，如果这些激活函数的偏导比较小（小于1）或者为0，那么随时间很容易发生梯度消失；相反，如果这些激活函数的偏导比较大（大于1），那么很有可能就会梯度爆炸。
    >
    > 
    >
    > 例如：小明是中国人，小明国家的首都是：_____。这时，RNN只需要参考横线前面的词，
    >
    > 最易懂的RNN的结构图：
    >
    > ![](imgs/deep_learning/RNN/RNN.png)
    >
    > 简单的RNN内部结构为：
    >
    > ![](imgs/deep_learning/RNN/RNN_small_structure.png)
    >
    > ![](imgs/deep_learning/RNN/fig1.png)
    >
    > ![](imgs/deep_learning/RNN/fig2.png)
    >
    > 但，当我们推测如下的：小明很喜欢吃蛋挞，所以小明下班后决定去商店__两个蛋挞。这时，不仅需要参考前面的词，还需要参考后面的词，才能推测出中间横线上的词，最大的概率是买。这就需要双向循环神经网络。如下图：
    >
    > ![](imgs/deep_learning/RNN/Bidirectional_recurrent_neural_network.png)
    >
    > 以及深层神经网络
    >
    > ![](imgs/deep_learning/RNN/Deep_Bidirectional_recurrent_neural_network.png)
    >
    > 具体公式实现可参考：https://zybuluo.com/hanbingtao/note/541458
    >
    > ![](imgs/deep_learning/RNN/fig3.png)
    >
    > * LSTM
    >
    >     * 前向计算：有三个门：遗忘门、输入门、输出门
    >
    >     >**遗忘门（forget gate）**，它决定了上一时刻的单元状态有多少保留到当前时刻；
    >     >
    >     >**输入门（input gate）**，它决定了当前时刻网络的输入有多少保存到单元状态。
    >     >
    >     >**输出门（output gate）**来控制单元状态有多少输出到LSTM的当前输出值。
    >     >
    >     >![](imgs/deep_learning/RNN/LSTM.png)
    >     >
    >     >![](imgs/deep_learning/RNN/fig4.png)
    >     >
    >     >
    >
    >     * 遗忘门
    >
    >     > sigmod函数，0代表舍弃，1代表保留。
    >     >
    >     > ![](imgs/deep_learning/RNN/forget_gate.png)
    >     >
    >     > ![](imgs/deep_learning/RNN/fig5.png)
    >
    >     * 输入门
    >
    >     >决定什么样的新信息被存放在细胞状态中。当前输入的单元状态通过 tanh 进行处理（得到一个在 -1 到 1 之间的值）。
    >     >
    >     >![](imgs/deep_learning/RNN/input_gate1.png)
    >     >
    >     >计算当前时刻的单元状态：它是由上一次的单元状态按元素乘以遗忘门，再用当前输入的单元状态按元素乘以输入门，再将两个积加和产生的，
    >     >
    >     >![当前细胞状态](imgs/deep_learning/RNN/input_gate2.png)
    >
    >     * 输出门
    >
    >     >LSTM最终的输出，是由输出门和单元状态共同确定的：把细胞状态通过 tanh 进行处理（得到一个在 -1 到 1 之间的值）并将它和 sigmoid 门的输出相乘，最终我们仅仅会输出我们确定输出的那部分。
    >     >
    >     >![](imgs/deep_learning/RNN/output_gate.png)
    >     >
    >     >
    >
    > * GRU
    >
    > > 省略了一个门，而且余下的两个门是z,1-z的关系。
    > >
    > > ![](imgs/deep_learning/RNN/fig7.png)
    > >
    > > ![](imgs/deep_learning/RNN/GRU.png)

* LSTM为什么可以解决梯度消失问题：

    > ![](imgs/deep_learning/RNN/fig6.png)

### 4. lstm中是否可以用relu作为激活函数？

> 在默认的情况下，LSTM使用tanh函数作为其激活函数。
>
> 看到当LSTM组成的神经网络层数比较少的时候，才用其默认饿tanh函数作为激活函数比Relu要好很多。
>
> 随着LSTM组成的网络加深，再继续使用tanh函数，就存在了梯度消失的的风险，导致一直徘徊在一个点无法搜索最优解，这种情况下，可以采用Relu函数进行调整，注意学习率需要变地更小一点防止进入死神经元。

### 5. lstm各个门分别使用什么激活函数？

* sigmoid 用在了各种gate上，产生0~1之间的值，这个一般只有sigmoid最直接了。
* tanh 用在了状态和输出上，是对数据的处理，这个用其他激活函数或许也可以



**LSTM各个门的作用？以及计算公式**

LSTM（Long Short-Term Memory）是一种常用的循环神经网络（RNN）模型，用于处理序列数据。

LSTM模型中包含三个门控单元（input gate，forget gate，output gate）和一个记忆单元（cell state），其作用如下：

输入门（input gate）：控制当前输入有多少信息应该被存储到cell state中。具体计算公式如下：

$it = \sigma(W{xi}xt + W{hi}h{t-1} + bi)$

其中，$it$表示输入门的输出，$\sigma$表示sigmoid激活函数，$W{xi}$、$W{hi}$和$bi$分别表示输入、上一时刻隐藏状态和偏置项对输入门的权重。

遗忘门（forget gate）：控制哪些过去的信息应该被遗忘。具体计算公式如下：

$ft = \sigma(W{xf}xt + W{hf}h{t-1} + bf)$

其中，$ft$表示遗忘门的输出，$\sigma$表示sigmoid激活函数，$W{xf}$、$W{hf}$和$bf$分别表示输入、上一时刻隐藏状态和偏置项对遗忘门的权重。

更新门（update gate）：控制哪些过去的信息应该被遗忘，并决定有哪些输入信息应该被添加到cell state中。具体计算公式如下：

$gt = \tanh(W{xg}xt + W{hg}h{t-1} + bg)$

$ut = \sigma(W{xu}xt + W{hu}h{t-1} + bu)$

其中，$gt$表示当前输入被加入到cell state的值，$\tanh$表示双曲正切函数，$ut$表示更新门的输出，$\sigma$表示sigmoid激活函数，$W{xg}$、$W{hg}$、$bg$、$W{xu}$、$W{hu}$和$bu$分别表示输入、上一时刻隐藏状态和偏置项对更新门和当前输入值的权重。

输出门（output gate）：控制哪些信息应该被输出到下一时刻。具体计算公式如下：

$ot = \sigma(W{xo}xt + W{ho}h{t-1} + bo)$

其中，$ot$表示输出门的输出，$\sigma$表示sigmoid函数，$W{xo}$、$W{ho}$和$bo$分别表示输入、上一时刻隐藏状态和偏置项对输出门的权重。

记忆单元（cell state）：cell state中保存了模型中的长期记忆信息，即当前时刻的输入信息是否需要被保存，以及前面时刻遗忘的部分是否需要被恢复。更新方式如下：

$Ct = ftC{t-1} + itgt$

其中，$Ct$表示当前时刻的cell state，$ft$表示遗忘门的输出，$C{t-1}$表示上一时刻的cell state，$it$表示输入门的输出，$gt$表示当前时刻的输入被加入到cell state的值。

