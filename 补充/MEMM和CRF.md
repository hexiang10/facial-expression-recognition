## 最大熵马尔科夫和CRF



最大熵模型（MaxEnt）：指的是多元逻辑回归

由于等概率的分布具有最大熵，所以最大熵的模型通过词性标注问题来描述就是：

1. 在没有任何假设的情况下，认为每种词性的概率都是相同的，假设有10中词性，那么每个词性的概率都是1/10
2. 如果语料表明，所有的词语出现的词性只有10个中的四个，那么此时，调整所有词的词性为$A:1/4 ,B:1/4,C:1.4,D:1/4,E:0....$
3. 当我们继续增加语料，发现A和B的概率很高，10次中有8次，某个词的词性不是A就是B，那么此时调整词性概率为：$A:4/10,B:4/10,C:1/10,D:1/10$
4. 重复上述过程

寻找一个熵最大的模型，就是要使用多元逻辑回归，训练他的权重w，让训练数据能够似然度最大化

> 训练数据能够似然度最大化：训练数据是总体的一个抽样，让训练数据尽可能能够代表总体，从而可以让模型可以有更好的表现力

**最大熵马尔科夫模型（MEMM）**是马尔科夫模型的变化版本。在马尔科夫模型中，我们使用贝叶斯理论来计算最有可能的观测序列，即：
$$
\hat{t}_n = \mathop{argmax}_{t_n}P(t_n|w_n) = \mathop{argmax}_{t_n}P(w_i|t_i)P(t_i|t_{i-1})
$$
但是在MEMM中，他直接去计算了后验概率P(t|w),直接对每个观测值的状态进行分类，在MEMM中，把概率进行了拆解：
$$
\hat{T} = \mathop{argmax}_T P(T|W) = \mathop{argmax}\prod_i P(tag_i|word_i,tag_{i-1})
$$
即:使用前一个状态tag和当前的词word，计算当前tag。

和隐马尔可夫模型不同的是，在上述的公式中，对于计算当前tag的分类过程中，输入不仅可以是$word_i和tag_{i-1}$,还可以包含其他的特征，比如：词语的第一个字母是否为大写，词语的后缀类型，前缀类型的等等。

所以MEMM的表现力会比HMM要更好。



## 条件随机场

**条件随机场(conditional random field,CRF)**是有输入x和输出y组成的一种无向图模型，可以看成是最大熵马尔可夫模型的推广。

下图是我们的常用于词性标注的线性链 条件随机场的图结构。其中x是观测序列，Y是标记序列

![](../../../../NLP%E8%AF%BE%E4%BB%B6%E7%BC%96%E5%86%99/markdown/doc/images/%E8%A1%A5%E5%85%85/%E6%9D%A1%E4%BB%B6%E9%9A%8F%E6%9C%BA%E5%9C%BA.png)

下图是HMM，MEMM，CRF的对比

![](../../../../NLP%E8%AF%BE%E4%BB%B6%E7%BC%96%E5%86%99/markdown/doc/images/%E8%A1%A5%E5%85%85/%E4%B8%8D%E5%90%8C%E5%AF%B9%E6%AF%94.png)



当观测序列为 $x=x_1,x_2...$ 时，状态序列为 $y=y_1,y_2....$的概率可写为:
$$
P(Y=y|x)=\frac{1}{Z(x)}\exp\biggl(\sum_k\lambda_k\sum_it_k(y_{i-1},y_i,x,i)+\sum_l\mu_l\sum_is_l(y_i,x,i)\biggr) \\
Z(x)=\sum_y\exp\biggl(\sum_k\lambda_k\sum_it_k(y_{i-1},y_i,x,i)+\sum_l\mu_l\sum_is_l(y_i,x,i)\biggr)
$$
其中$Z(x)$是归一化因子，类似softmax中的分母，计算的是所有可能的y的和

后面的部分由**特征函数**组成：

**转移特征：** $t_k(y_{i-1},y_i,x,i)$ 是定义在边上的特征函数（transition），依赖于当前位置 i 和前一位置 i-1 ；对应的权值为 $\lambda_k$ 。

**状态特征：** $s_l(y_i,x,i)$ 是定义在节点上的特征函数（state），依赖于当前位置 i&nbsp;；对应的权值为 $\mu_l$ 。

一般来说，特征函数的取值为 1 或 0 ，当满足规定好的特征条件时取值为 1 ，否则为 0 。

对于`北\B京\E欢\B迎\E你\E`特征函数可以如下：

```
func1 = if (output = B and feature="北") return 1 else return 0
func2 = if (output = M and feature="北") return 1 else return 0
func3 = if (output = E and feature="北") return 1 else return 0
func4 = if (output = B and feature="京") return 1 else return 0
```

每个特征函数的权值 类似于发射概率，是统计后的概率。

