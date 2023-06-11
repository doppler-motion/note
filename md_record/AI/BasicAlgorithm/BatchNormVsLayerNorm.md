### 26. BN的原理、作用；手动推导

>   参考：[(3条消息) BatchNormalization、LayerNormalization、InstanceNorm、GroupNorm、SwitchableNorm总结_夏洛的网的博客-CSDN博客](https://blog.csdn.net/liuxiao214/article/details/81037416)
>
>   [深度学习中的五种归一化（BN、LN、IN、GN和SN）方法简介_in归一化_时光碎了天的博客-CSDN博客](https://blog.csdn.net/u013289254/article/details/99690730)

#### 原理

#### 作用

*   加快网络的训练速度与收敛的速度
*   控制梯度爆炸防止梯度消失
*   防止过拟合

### 27. BN在前向传播和后向传播中的区别

>   BN（Batch Normalization）是一种常用的神经网络优化技术，用于加速深度神经网络的训练并提高模型的准确性。BN在前向传播和反向传播中的作用是不同的。
>   在前向传播期间，BN计算每个batch的均值和方差，并标准化输入数据。标准化后的数据可以使后续层的输入在同一分布上，并加速网络的训练。BN公式如下：
>   $$
>   \hat{x_{i}}=\frac{x_{i}-\mu B}{\sqrt{\sigma B^2+\epsilon}}
>   $$
>   其中，$x_{i}$是输入数据中的第$i$个元素，$\mu B$和$\sigma B^2$分别是当前batch中所有$xi$的均值和方差，$\epsilon$是一个小常数，以避免除以0的情况。最后，将标准化后的数据进行缩放和平移，得到输出：
>   $$
>   y_{i}=\gamma\hat{x_{i}}+\beta
>   $$
>   其中，$\gamma$和$\beta$是可学习的参数，用于缩放和平移标准化后的数据，以恢复网络的表达能力。
>   在反向传播期间，BN的作用是计算梯度以更新参数。具体来说，需要计算参与标准化、缩放和平移的三个变量$\hat{x}$、$\gamma$和$\beta$的偏导数。
>   求$\hat{x}$的偏导数时，需要计算对$\mu B$和$\sigma B^2$的偏导数。对于$\gamma$和$\beta$，可以按照通常的反向传播方法计算它们的偏导数。最终，可以使用梯度下降等优化算法来更新BN中的参数。