# Python编程实现线性判别分析LDA
---
# 什么是线性判别分析

## 投影后类内方差最小，类间方差最大

线性判别分析（Linear Discriminant Analysis, 以下简称LDA），是一种监督学习的降维技术，LDA的思想可以用一句话概括，就是“投影后类内方差最小，类间方差最大”

> 方差可以理解为聚集的程度，方差越小，投影越靠拢

如下图的例子：
![在这里插入图片描述](fef23330b6714349be03b3d98e50df75.png)

从直观上可以看出，右图要比左图的投影效果好，因为右图的黑色数据和蓝色数据各个较为集中，且类别之间的距离明显。左图则在边界处数据混杂。以上就是LDA的主要思想了

## 阈值

我们找到这样一个投影后，再在这个投影上找到一个阈值，即分为两类概率相等的地方。在这个阈值的两侧就分类为这不同的两类
![在这里插入图片描述](39ff5fb8b1a04c239f4d4f65460ec1f5.png)


> 参考资料：
>
> https://www.bilibili.com/video/BV175411W7hv
>
> https://www.cnblogs.com/pinard/p/6244265.html

# 数学基础

## 共轭转置矩阵与Hermitan矩阵

共轭转置矩阵$\mathbf{A}^H$是在$\mathbf{A}^\intercal$的基础上将每一个元素取共轭，而Hermitan矩阵就是$\mathbf{A}^H=\mathbf{A}$的矩阵

## **瑞利商（Rayleigh quotient）**

定义：

$$R(\mathbf{A},\mathbf{x})=\frac{\mathbf{x}^H\mathbf{A}\mathbf{x}}{\mathbf{x}^H\mathbf{x}}$$

其中，$\mathbf{x}$为非零向量，$\mathbf{A}$为Hermitan矩阵

性质：

1. 它可以提供方阵$\mathbf{A}$的最大和最小特征值的一个界

它的最大值等于矩阵$\mathbf{A}$最大的特征值，而最小值等于矩阵$\mathbf{A}$的最小的特征值，即满足：

$$\lambda_{min}<R(\mathbf{A},\bm{x})<\lambda_{max}$$

当向量$\mathbf{x}$是标准正交基时，即$\mathbf{x}^H\mathbf{x}=1$，瑞利商退化为$R(\mathbf{x},\mathbf{x})={\mathbf{x}^H}\mathbf{A}\mathbf{x}$

> 瑞利商迭代法是一种寻找矩阵特征值的有效方法

1. 如果$\mathbf{x}$是$\mathbf{A}$的一个特征向量，那么$R(\mathbf{A},\mathbf{x})$正好是对应的特征值

## **广义瑞利商（genralized Rayleigh quotient）**

定义：

$$R(\mathbf{A},\mathbf{B},\mathbf{x})=\frac{{\mathbf{x}^H}\mathbf{A}\mathbf{x}}{{\mathbf{x}^H}\mathbf{B}\mathbf{x}}$$

其中，$\mathbf{x}$为非零向量，$\mathbf{A},\mathbf{B}$为Hermitan矩阵，$\mathbf{B}$为正定矩阵

性质：

它的最大值等于矩阵$\mathbf{B}^{-1}\mathbf{A}$最大的特征值，而最小值等于矩阵$\mathbf{B}^{-1}\mathbf{A}$的最小的特征值

> 此处，设$\mathbf{x}=\mathbf{B}^{-\frac{1}{2}}\mathbf{x}'$，将广义瑞利商变换为瑞利商，再利用瑞利商的性质可以得到

## 协方差与协方差矩阵

### 协方差

定义：

$$cov(a,b)=\frac{1}{m-1}\sum_{i=1}^{m}(a_i-\mu_a)(b_i-\mu_b)$$

其中，$m$是样本数，$a$与$b$是两个特征，$\mu_a$与$\mu_b$是两个特征的平均值

> - 分母是m-1的情况下，估计值是总体方差的无偏估计
> - 分母是m的情况下，值是最大似然估计
> - 分母是m+1的情况下，值是最小MSE（Mean Squared Error) 的估计
>
> 此处不展开
> ![在这里插入图片描述](fc2737e536e8496e8e9d9739eb1a4299.png)
### 协方差矩阵

定义：

$$\bf{\Sigma}_j=(\mathbf{x}-\mathbf{\mu}_j)(\mathbf{x}-\mathbf{\mu}_j)^\intercal$$

> $\mathbf{x}$和$\mathbf{\mu}_j$都为$n$维列向量
>
> $\bf{\Sigma}_j$为$n\times{n}$矩阵
>
> ![在这里插入图片描述](2b086797399543e7accfcb23d9b48010.png)

对角线上是方差，其他元素是协方差

> 参考资料：
>
> https://www.bilibili.com/video/BV1D84y1m7rj/

## 拉格朗日乘子法

一种寻找多元函数在一组约束下的极值的方法，通过引入拉格朗日乘子，将有$d$个变量与$k$个约束条件的最优化问题转换为具有$d+k$个变量的无约束优化问题求解

### 有一个等式约束条件的最优化问题

通常，一个有$d$个变量与一个等式约束条件的最优化问题是这样表述的：

假定$\mathbf{x}$是$d$维向量，需要找到$\mathbf{x}^*$使得目标函数$f(\mathbf{x})$最小，并满足一个等式约束$g(\mathbf{x})=0$

这个问题的几何意义是在方程$g(\mathbf{x})=0$确定的$d-1$维曲面上寻找到使得$f(\mathbf{x})$最小的点

> 一个等式约束降低一个维度

由此可以得出下面两个结论：

1. 对于约束曲面上的任意点$\mathbf{x}$，该点的梯度$\nabla g(\mathbf{x})$正交与约束曲面
2. 在最优点$\mathbf{x}^*$，目标函数在该点的梯度$\nabla f(\mathbf{x})$正交于约束曲面

### 约束最优化问题转换为无约束优化问题

所以在最优点$\mathbf{x}^*$，$\nabla g(\mathbf{x})$与$\nabla f(\mathbf{x})$一定方向相同或相反，也就是存在$\lambda \neq 0$，使得：

$$\nabla f(\mathbf{x})+\lambda \nabla g(\mathbf{x}) =0$$

$\lambda$称为拉格朗日乘子，我们定义拉格朗日函数：

$$L(\mathbf{x},\lambda)=f(\mathbf{x})+\lambda g(\mathbf{x})$$

对于拉格朗日函数，有：

$$\begin{cases} \nabla_x \ L(\mathbf{x},\lambda) =\nabla f(\mathbf{x})+\lambda\nabla g(\mathbf{x}) =0 \\ \nabla_\lambda\ L(\mathbf{x},\lambda)=g(\mathbf{x})=0 \end{cases} $$

> 式子一为前面推导的$\nabla g(\mathbf{x})$与$\nabla f(\mathbf{x})$一定方向相同或相反
>
> 式子二为约束条件

最优化问题就转换为了为对拉格朗日函数$L(\mathbf{x},\lambda)$的无约束优化问题

## 矩阵求导

$\mathbf{w}^\intercal A\mathbf{w}$对$\mathbf{w}$求偏导，得到的结果是$(A+A^\intercal)\mathbf{w}$

# **二类LDA原理**

二类指的是样本可以被分为两类，而样本的特征都为$n$维向量

## 相关定义

假设我们的数据集

$$D=\{(\mathbf{x}_1,y_1),(\mathbf{x}_2,y_2),...,((\mathbf{x}_m,y_m))\}$$

其中任意样本$\mathbf{x}_i$为$n$维向量，$y_i\in\{0,1\}$

我们定义:

1. 第$j$类样本的个数为：$N_j(j=0,1)$
2. 第$j$类样本的集合为：$X_j(j=0,1)$
3. 第$j$类样本的均值向量为：

$$\mathbf{\mu}_j(j=0,1)=\frac{1}{N_j}\sum_{\mathbf{x}{\in}X_j}\mathbf{x}(j=0,1)$$

> $\mathbf{\mu}_j(j=0,1)$是$n$维列向量

4. 第$j$类样本的协方差矩阵:
$$\mathbf{\Sigma}_j=\frac{\sum_{{\mathbf{x}\in}X_j}(\mathbf{x}-\mathbf{\mu}_j)(\mathbf{x}-\mathbf{\mu}_j)^\intercal}{N_j-1}(j=0,1)$$

> $\mathbf{\Sigma}_j$为$n{\times}n$矩阵

## 投影

### 相关数学原理

#### 投影计算

假设我们的投影直线是向量$\mathbf{w}$,则对任意一个样本$\mathbf{x}_i$,它在直线$\mathbf{w}$的投影为$\mathbf{w}^{\intercal}\mathbf{x}_i$

> 点积，$\mathbf{w}^{\intercal}\mathbf{x}_i={\lvert}\mathbf{w}{\rvert}{\lvert}\mathbf{x}_i{\rvert}cos(\theta)$

#### L-P范数

范数（Norm），是具有“长度”概念的函数，而L-P范数是其中一种，它是这么定义的：

$$L_p=\lVert{x}\rVert_p=(\sum_{i=1}^n{x_i^p})^\frac{1}{p}$$

性质：

1. 正无穷范数等价于$max(x_i)$
2. 负无穷范数等价于$min(x_i)$

## 二类LDA原理

### 类间方差最大

由于是两类数据，因此我们只需要将数据投影到一条直线上即可。

对于我们的两个类别的中心点$\mathbf{\mu}_0,\mathbf{\mu}_1$，在直线$\mathbf{w}$的投影为$\mathbf{w}^{\intercal}\mathbf{\mu}_0$和$\mathbf{w}^{\intercal}\mathbf{\mu}_1$

LDA需要让不同类别的数据的类别中心之间的距离尽可能的大

也就是我们要最大化：

$${\lVert}\mathbf{w}^\intercal\mathbf{\mu}_0−\mathbf{w}^\intercal\mathbf{\mu}_1{\rVert}^2_2$$

> L2范数再求平方

### 类内方差最小

同时我们希望同一种类别数据的投影点尽可能的接近，也就是令

$${\sum_{{\mathbf{x}\in}X_j} \lVert{}(\mathbf{w}^{\intercal}\mathbf{x}-\mathbf{w}^\intercal\mathbf{\mu}_j)} \rVert_2^2$$

尽可能的小

> $\mathbf{w}^{\intercal}\mathbf{x}(\mathbf{x}\in{X_j})$是单个样本在$\mathbf{w}$上的投影
>
> $\mathbf{w}^\intercal\mu_0$是样本均值在$\mathbf{w}$上的投影

$$\begin{equation*} \begin{split} &{\sum_{{\mathbf{x}\in}X_j}\lVert{}(\mathbf{w}^{\intercal}\mathbf{x}-\mathbf{w}^\intercal\mathbf{\mu}_j)}\rVert_2^2\\  =&{\sum_{{\mathbf{x}\in}X_j}}(\mathbf{w}^{\intercal}(\mathbf{x}-\mathbf{\mu}_j))^2\\  =&{\sum_{{\mathbf{x}\in}X_j}}(\mathbf{w}^{\intercal}(\mathbf{x}-\mathbf{\mu}_j))(\mathbf{w}^{\intercal}(\mathbf{x}-\mathbf{\mu}_j))^{\intercal} \\  =&{\sum_{{\mathbf{x}\in}X_j}}(\mathbf{w}^{\intercal}(\mathbf{x}-\mathbf{\mu}_j))(\mathbf{w}^{\intercal}(\mathbf{x}-\mathbf{\mu}_j))^{\intercal}\\  =&{\sum_{{\mathbf{x}\in}X_j}}(\mathbf{w}^{\intercal}(\mathbf{x}-\mathbf{\mu}_j)(\mathbf{x}-\mathbf{\mu}_j)^{\intercal}\mathbf{w})\\  =&{\sum_{{\mathbf{x}\in}X_j}}\mathbf{w}^{\intercal}\mathbf{\Sigma}_j\mathbf{w}(j=0,1) \end{split} \end{equation*}$$

> 因为$\mathbf{w}^{\intercal}(\mathbf{x}-\mathbf{\mu}_j)$是常量，所以可以整体取转置，不影响结果，便于计算

也就是要同类样本投影点的协方差$\mathbf{w}^\intercal{\mathbf{\Sigma}}_0\mathbf{w}$和$\mathbf{w}^\intercal{\mathbf{\Sigma}}_1\mathbf{w}$尽可能的小

即最小化：

$$\mathbf{w}^\intercal{\mathbf{\Sigma}}_0\mathbf{w} +\mathbf{w}^\intercal{\mathbf{\Sigma}}_1\mathbf{w}$$

### 优化目标，类内散度矩阵与类间散度矩阵

综上所述，我们可以将优化目标定义为：

$$\begin{split} \underbrace{\arg\max}_{\mathbf{w}}J(\mathbf{w}) &=\frac{{\lVert}\mathbf{w}^\intercal\mathbf{\mu}_0−\mathbf{w}^\intercal\mathbf{\mu}_1{\rVert}^2_2} {\mathbf{w}^\intercal\mathbf{\Sigma}_0\mathbf{w}+\mathbf{w}^\intercal\mathbf{\Sigma}_1\mathbf{w}}\\ &=\frac{\mathbf{w}^\intercal(\mathbf{\mu}_0−\mathbf{\mu}_1)(\mathbf{\mu}_0−\mathbf{\mu}_1)^{\intercal}\mathbf{w}}{\mathbf{w}^\intercal(\mathbf{\Sigma}_0+\mathbf{\Sigma}_1)\mathbf{w}} \end{split}$$

> $\underbrace{\arg\max}_{\mathbf{w}}J(\mathbf{w})$表示令函数$J(\mathbf{w})$最大的时候的$\mathbf{w}$值

这个表达式的形式就和广义瑞利商很相近了，我们可以定义：

1. 类内散度矩阵：$\mathbf{S}_w=\bf{\Sigma}_0+\bf{\Sigma}_1$
2. 类间散度矩阵：$\mathbf{S}_b=(\mathbf{\mu}_0−\mathbf{\mu}_1)(\mathbf{\mu}_0−\mathbf{\mu}_1)^{\intercal}$

> $\mathbf{S}_w$和$\mathbf{S}_b$都是对称矩阵

此时优化目标可以写为：

$$\begin{split} \underbrace{\arg\max}_{\mathbf{w}}J(\mathbf{w}) &=\frac{\mathbf{w}^\intercal\mathbf{S}_b\mathbf{w}} {\mathbf{w}^\intercal{\mathbf{S}_w}\mathbf{w}}\\ &=R(\mathbf{S}_b,\mathbf{S}_w,\mathbf{w}) \end{split}$$

利用广义瑞利商的最大值等于矩阵$\mathbf{B}^{-1}\mathbf{A}$最大的特征值，而最小值等于矩阵$\mathbf{B}^{-1}\mathbf{A}$的最小的特征值的性质

> $R(\mathbf{A},\mathbf{B},\mathbf{x})=\frac{{\mathbf{x}^H}\mathbf{A}\mathbf{x}}{{\mathbf{x}^H}\mathbf{B}\mathbf{x}}$

$J(\mathbf{w})$的最大值为矩阵$\mathbf{S}_w^{-1}\mathbf{S}_b$的最大特征值，对应的$\mathbf{w}$就是该特征值对应的特征向量

## 使用拉格朗日乘子法确定投影直线

$$\begin{split} \underbrace{\arg\max}_{\mathbf{w}}J(\mathbf{w}) &=\frac{\mathbf{w}^\intercal{\mathbf{S}_b}\mathbf{w}}{\mathbf{w}^\intercal{\mathbf{S}_w}\mathbf{w}}\\ &=R(\mathbf{S}_b,\mathbf{S}_w,\mathbf{w}) \end{split}$$

我们可以发现，式子中分子分母都是关于$\mathbf{w}$的二次项，所以$\mathbf{w}^*$和$a\mathbf{w}^*$都是解。也就是说，解与$\mathbf{w}$的长度无关，只与其方向有关

所以，我们可以假设$\mathbf{w}^\intercal{\mathbf{S}_w}\mathbf{w}=1$，将目标转化为：

$$\begin{split} {\underbrace{\arg\min}_{\mathbf{w}}} &\quad{-\mathbf{w}^\intercal{\mathbf{S}_b}\mathbf{w}}\\ s.t. &\quad\mathbf{w}^\intercal{\mathbf{S}_w}\mathbf{w}=1 \end{split}$$

由拉格朗日乘子法，可设拉格朗日函数为：

$$\begin{split} L(\mathbf{w},\lambda) &=-\mathbf{w}^\intercal \mathbf{S}_b \mathbf{w} +\lambda(\mathbf{w}^\intercal{\mathbf{S}_w}\mathbf{w}-1)\\ \nabla_wL(\mathbf{w},\lambda) &=-2\mathbf{S}_b\mathbf{w}+2\lambda\mathbf{S}_w\mathbf{w} \end{split}$$

> $\mathbf{S}_w$和$\mathbf{S}_b$都是对称矩阵

又：

$$\begin{split} \nabla_wL(\mathbf{w},\lambda)& =0\\ -2\mathbf{S}_b\mathbf{w}+2\lambda\mathbf{S}_w\mathbf{w}& =0\\ \mathbf{S}_b\mathbf{w}& =\lambda\mathbf{S}_w\mathbf{w}\\ (\mathbf{\mu}_0−\mathbf{\mu}_1)(\mathbf{\mu}_0−\mathbf{\mu}_1)^{\intercal}\mathbf{w}& =\lambda\mathbf{S}_w\mathbf{w} \end{split}$$

我们令$(\mathbf{\mu}_0−\mathbf{\mu}_1)^{\intercal}\mathbf{w}=\gamma$，得到：

$$\begin{split} \gamma(\mathbf{\mu}_0−\mathbf{\mu}_1)& =\lambda\mathbf{S}_w\mathbf{w}\\ \mathbf{w}& =\frac{\gamma}{\lambda}\mathbf{S}_w^{-1}(\mathbf{\mu}_0−\mathbf{\mu}_1) \end{split}$$

> $\lambda$、$\gamma$是常量

因为我们不关心$\mathbf{w}$的大小，只关心方向，所以我们可以求：

$\mathbf{w}' =\frac{\lambda}{\gamma}\mathbf{w} =\mathbf{S}_w^{-1}(\mathbf{\mu}_0−\mathbf{\mu}_1)$

> 其中：
>
> $\mathbf{S}_w =\bf{\Sigma}_0+\bf{\Sigma}_1 =(\mathbf{x}-\mathbf{\mu}_0)(\mathbf{x}-\mathbf{\mu}_0)^\intercal+(\mathbf{x}-\mathbf{\mu}_1)(\mathbf{x}-\mathbf{\mu}_1)^\intercal$
>
> $\mathbf{\mu}_j(j=0,1)=\frac{1}{N_j}\sum_{\mathbf{x}{\in}X_j}\mathbf{x}(j=0,1)$

即得到了投影直线$\mathbf{w}$

# 例题：

## 题目

请用Python编程实现线性判别分析LDA，并给出下面数据集上的结果及说明。如果你在作业完成过程当中借助了ChatGPT等AI工具，请写出相应的使用过程说明。

| 编号 | 属性1 | 属性2 | 类别 |
| ---- | ----- | ----- | ---- |
| 1    | 0.666 | 0.091 | 正例 |
| 2    | 0.243 | 0.267 | 正例 |
| 3    | 0.244 | 0.056 | 正例 |
| 4    | 0.342 | 0.098 | 正例 |
| 5    | 0.638 | 0.16  | 正例 |
| 6    | 0.656 | 0.197 | 正例 |
| 7    | 0.359 | 0.369 | 正例 |
| 8    | 0.592 | 0.041 | 正例 |
| 9    | 0.718 | 0.102 | 正例 |
| 10   | 0.697 | 0.46  | 反例 |
| 11   | 0.774 | 0.376 | 反例 |
| 12   | 0.633 | 0.263 | 反例 |
| 13   | 0.607 | 0.317 | 反例 |
| 14   | 0.555 | 0.214 | 反例 |
| 15   | 0.402 | 0.236 | 反例 |
| 16   | 0.481 | 0.149 | 反例 |
| 17   | 0.436 | 0.21  | 反例 |
| 18   | 0.557 | 0.216 | 反例 |

## 代码

```Python
# 2152952 自动化 孔维涛
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Hei'] # 设置中文显示字体
plt.rcParams['axes.unicode_minus'] = False   # 正确显示负号

# 数据集
data = np.array([
    [0.666, 0.091, 1],
    [0.243, 0.267, 1],
    [0.244, 0.056, 1],
    [0.342, 0.098, 1],
    [0.638, 0.16, 1],
    [0.656, 0.197, 1],
    [0.359, 0.369, 1],
    [0.592, 0.041, 1],
    [0.718, 0.102, 1],
    [0.697, 0.46, 0],
    [0.774, 0.376, 0],
    [0.633, 0.263, 0],
    [0.607, 0.317, 0],
    [0.555, 0.214, 0],
    [0.402, 0.236, 0],
    [0.481, 0.149, 0],
    [0.436, 0.21, 0],
    [0.557, 0.216, 0]
])

# 分离特征和标签
X = data[:, :2]
y = data[:, 2]

# 计算正例和反例的均值向量
mu_0 = np.mean(X[y == 0], axis=0)
mu_1 = np.mean(X[y == 1], axis=0)

# 计算类内散度矩阵
S_w = np.zeros((2, 2))
for i in range(len(X)):
    if y[i] == 0:
        S_w += np.outer(X[i] - mu_0, X[i] - mu_0)
    else:
        S_w += np.outer(X[i] - mu_1, X[i] - mu_1)

# 计算最优投影方向
w_prime = np.linalg.inv(S_w).dot(mu_0 - mu_1)

# 计算LDA分割线（决策边界），决策边界垂直于投影向量w_prime，且通过两个类别均值的中点
mid_point = (mu_0 + mu_1) / 2
slope = w_prime[1] / w_prime[0]

# 计算垂直于投影向量的斜率
slope_perpendicular = -1 / slope

# 计算截距
intercept_perpendicular = mid_point[1] - slope_perpendicular * mid_point[0]

# 绘制数据点
plt.figure(figsize=(10, 6))
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='正例')
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='反例')

# 绘制LDA分割线
x_values = np.linspace(0, 1, 100)  # x值
y_values = slope * x_values + intercept_perpendicular  # 对应的y值
plt.plot(x_values, y_values, color='green', label='LDA 分割线')

# 其他图形设置
plt.xlabel('属性1')
plt.ylabel('属性2')
plt.legend()
plt.title('数据集和LDA分割线')
plt.show()
```

## 结果

```Python
投影向量为：[0.14340516 0.68106694]
LDA分割线的方程为：y = -0.21055955976278984x + 0.3246317652068213
```
![在这里插入图片描述](e2c0b4b7eb994586a7a14c4e73096391.png)


## 代码解析

### 导入包

```Python
import numpy as np
import matplotlib.pyplot as plt
```

NumPy 是 Python 中用于科学计算的核心库之一，提供了高性能的多维数组对象以及用于数组操作的各种工具

Matplotlib 是 Python 中用于创建静态、交互式和动态可视化的二维图形库。它提供了类似于 MATLAB 的绘图接口，使得用户能够轻松地生成各种类型的图表、图形和动画

### Matplotlib设置字体

```Python
plt.rcParams['font.sans-serif'] = ['Hei'] # 设置中文显示字体
plt.rcParams['axes.unicode_minus'] = False   # 正确显示负号
```

使用 Matplotlib 库中的 rcParams 属性，用于配置全局的绘图参数

1. 使用黑体，解决中文显示问题
2. 使用减号显示负号，解决显示问题

> - 在 Python 中，字典是一种无序的数据结构，用于存储键值对字典的语法是使用大括号 `{}` 来创建，键值对之间使用冒号 `:` 分隔，每个键值对之间使用逗号 `,` 分隔。`{'font.sans-serif': ['Hei']}` 是一个字典，其中 `'font.sans-serif'` 是键，`['Hei']` 是与之对应的值
> - `['Hei']`是一个列表，`plt.rcParams['font.sans-serif']` 这个参数要求接受一个字体名称的列表，而不是单独的字符串。虽然我们只需要指定一个中文字体，但是Matplotlib中 `font.sans-serif` 这个参数的设计是允许用户指定多个备选的字体名称的，这样做的目的是为了在系统中找不到指定字体时能够自动回退到备选的字体。因此，即使只有一个字体，也需要将其放入列表中作为 `font.sans-serif` 的值。
> - `plt.rcParams['axes.unicode_minus'] = False`：这一行代码用于设置图表中显示的负号（即减号）的样式，将其设置为不显示负号。在一些情况下，默认情况下 Matplotlib 可能会显示不正确的负号，比如显示为方块或者其他字符，通过将 `axes.unicode_minus` 参数设置为 False，可以避免这种情况设置为 False 表示不显示负号，而如果设置为 True，则显示标准的减号

### 设置数据集

```Python
# 数据集
data = np.array([
    [0.666, 0.091, 1],
    [0.243, 0.267, 1],
    [0.244, 0.056, 1],
    [0.342, 0.098, 1],
    [0.638, 0.16, 1],
    [0.656, 0.197, 1],
    [0.359, 0.369, 1],
    [0.592, 0.041, 1],
    [0.718, 0.102, 1],
    [0.697, 0.46, 0],
    [0.774, 0.376, 0],
    [0.633, 0.263, 0],
    [0.607, 0.317, 0],
    [0.555, 0.214, 0],
    [0.402, 0.236, 0],
    [0.481, 0.149, 0],
    [0.436, 0.21, 0],
    [0.557, 0.216, 0]
])
```

### 分离特征和标签

```Python
# 分离特征和标签
X = data[:, :2]
y = data[:, 2]
```

1. `X = data[:, :2]`

这行代码的作用是提取 `data` 数组中的前两列，这两列包含了每个数据点的属性

> - `data[:, :2]` 中的第一个 `:` 代表选择所有行
> - `:2` 表示选择从第一列开始到第二列（不包括索引为2的第三列）的所有列
>
> 结果 `X` 就是一个只包含数据集中所有数据点的属性的二维数组

1. `y = data[:, 2]`

这行代码的作用是提取 `data` 数组中的第三列，这一列包含了每个数据点的类别标签

> - `data[:, 2]` 中的第一个 `:` 同样代表选择所有行
> - `2` 表示选择第三列（因为在Python中索引是从0开始的）

结果 `y` 就是一个一维数组，包含了数据集中所有数据点的类别标签

### 计算均值向量

```Python
# 计算正例和反例的均值向量
mu_0 = np.mean(X[y == 0], axis=0)
mu_1 = np.mean(X[y == 1], axis=0)
```

1. `mu_0 = np.mean(X[y == 0], axis=0)`

这行代码计算所有类别标签为0的样本点的特征均值

`X[y == 0]` 是利用布尔索引从数组 `X` 中选择出所有类别标签为0的样本点。

> - `y == 0` 生成一个布尔数组，其中 `y` 中每个元素等于0的位置是 `True`，不等于0的位置是 `False`
> - 当这个布尔数组用作 `X` 的索引时，它会选择 `X` 中所有对应于 `True` 的行
> - `np.mean(..., axis=0)` 计算所选样本点特征的均值，`axis=0`指的是沿着数组的第一个轴进行操作，即指定沿着列（垂直方向）进行计算，即分别为每个特征（列）计算均值

1. `mu_1 = np.mean(X[y == 1], axis=0)`

> 与上一行代码类似，这行代码计算所有类别标签为1的样本点的特征均值。`X[y == 1]` 选择出了所有类别标签为1的样本点。

### 计算类内散度矩阵

```Python
# 计算类内散度矩阵
S_w = np.zeros((2, 2))
for i in range(len(X)):
    if y[i] == 0:
        S_w += np.outer(X[i] - mu_0, X[i] - mu_0)
    else:
        S_w += np.outer(X[i] - mu_1, X[i] - mu_1)
```

`np.outer()`函数计算了两个向量的外积，即将第一个向量乘以第二个向量的转置，产生一个矩阵。在这里，它被用来计算差向量乘以其转置，然后累加到类内散度矩阵中

> - 类间散度矩阵：
> $$\mathbf{S}_w =\bf{\Sigma}_0+\bf{\Sigma}_1 =\sum_{\mathbf{x}\in\mathbf{X}_0}(\mathbf{x}-\mathbf{\mu}_0)(\mathbf{x}-\mathbf{\mu}_0)^\intercal +\sum_{\mathbf{x}\in\mathbf{X}_1}(\mathbf{x}-\mathbf{\mu}_1)(\mathbf{x}-\mathbf{\mu}_1)^\intercal$$
>
> - 外积（Outer Product），也称为张量积或叉积，在向量的情况下，外积用于计算两个向量的结果矩阵，结果矩阵的维度等于输入向量的维度的乘积
> - 内积（Inner Product），也称为点积或数量积，在向量的情况下，内积用于计算两个向量之间的数量关系，结果是一个标量

### 计算最优投影方向

```Python
# 计算最优投影方向
w_prime = np.linalg.inv(S_w).dot(mu_0 - mu_1)
```

1. `np.linalg.inv(S_w)`: 这部分使用 NumPy 中的 `np.linalg.inv()` 函数计算了矩阵 `S_w` 的逆矩阵。`np.linalg.inv()` 用于计算给定矩阵的逆矩阵
2. `.dot(mu_0 - mu_1)`: 这部分对计算得到的逆矩阵与向量 `mu_0 - mu_1` 进行点乘操作。`mu_0 - mu_1` 表示两个均值向量的差向量，点乘操作则表示矩阵与向量的乘法运算

> 最优投影方向：$\mathbf{w}' =\frac{\lambda}{\gamma}\mathbf{w} =\mathbf{S}_w^{-1}(\mathbf{\mu}_0−\mathbf{\mu}_1)$

### 计算LDA分割线

```Python
# 计算LDA分割线（决策边界），决策边界垂直于投影向量w_prime，且通过两个类别均值的中点
mid_point = (mu_0 + mu_1) / 2
slope = w_prime[1] / w_prime[0]

# 计算垂直于投影向量的斜率
slope_perpendicular = -1 / slope

# 计算截距
intercept_perpendicular = mid_point[1] - slope_perpendicular * mid_point[0]
```

> 相互垂直的向量斜率之积为-1

### 绘制数据点

```Python
# 绘制数据点
plt.figure(figsize=(10, 6))
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='正例')
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='反例')
```

1. `plt.figure(figsize=(10, 6))`

这一行创建了一个新的图形窗口，其大小为10英寸宽和6英寸高。`figsize`参数是一个元组，指定了窗口的宽度和高度（单位是英寸）。这为接下来的绘图提供了一个“画布”

1. `plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='正例')`

这一行绘制了所有正例的数据点。`plt.scatter`函数用于绘制散点图，其中第一个参数是x轴上的值，第二个参数是y轴上的值

> `X[y == 1][:, 0]`选择了`X`中所有标签`y`为1的行的第一列（即第一个特征），用作x轴的坐标
>
> 1. `X[y == 1][:, 1]`选择了相同行的第二列（即第二个特征），用作y轴的坐标
>
> 1. `color='blue'`指定了点的颜色为蓝色
>
> 1. `label='正例'`给这些点的集合添加了一个标签，称为“正例”，这个标签将在图例中显示

1. `plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='反例')`

1. 这一行与前一行类似，但它是为所有反例的数据点绘制散点图

1. 通过`X[y == 0][:, 0]`和`X[y == 0][:, 1]`选择了所有标签`y`为0的数据点，分别用作x轴和y轴的坐标。

1. 这些点被设置为红色（`color='red'`），并被标记为“反例”（`label='反例'`）

### 绘制LDA分割线

```Python
# 绘制LDA分割线
x_values = np.linspace(0, 1, 100)  # x值
y_values = slope * x_values + intercept_perpendicular  # 对应的y值
plt.plot(x_values, y_values, color='green', label='LDA 分割线')
```

### 其他图形设置

```Python
# 其他图形设置
plt.xlabel('属性1')
plt.ylabel('属性2')
plt.legend()
plt.title('数据集和LDA分割线')
plt.show()
```

`plt.legend()` 用于添加图例到当前图形中
