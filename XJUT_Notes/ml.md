# 机器学习概论

## 数学知识复习

### 矩阵求导

$$
\begin{array}{l}\frac{\partial \mathbf{x}^{T} \mathbf{a}}{\partial \mathbf{x}}=\frac{\partial \mathbf{a}^{T} \mathbf{x}}{\partial \mathbf{x}}=\mathbf{a} \\ \frac{\partial \mathbf{a}^{T} \mathbf{X} \mathbf{b}}{\partial \mathbf{X}}=\mathbf{a b}^{T} \\ \frac{\partial \mathbf{a}^{T} \mathbf{X}^{T} \mathbf{b}}{\partial \mathbf{X}}=\mathbf{b a}^{T} \\ \frac{\partial \mathbf{a}^{T} \mathbf{X} \mathbf{a}}{\partial \mathbf{X}}=\frac{\partial \mathbf{a}^{T} \mathbf{X}^{T} \mathbf{a}}{\partial \mathbf{X}}=\mathbf{a a}^{T} \\ \frac{\partial \mathbf{x}^{T} \mathbf{B} \mathbf{x}}{\partial \mathbf{x}}=\left(\mathbf{B}+\mathbf{B}^{T}\right) \mathbf{x}\end{array}
$$

![机器学习.](/Users/wangjing/Downloads/机器学习..png)

# 多元线性回归

### 函数模型

**函数形式**
$$
f(x)=\theta_{0}+\theta_{1} x_{1}+\cdots+\theta_{p} x_{p}
$$
**向量形式**：

通常一个向量指的都是列向量，向量的转置是行向量
$$
f(x)=\sum_{i=0}^{p} \theta_{i} x_{i}=\boldsymbol{\theta}^{T} x=x^{T} \boldsymbol{\theta} = \left[\left(x_{0}=1\right), x_{1}, x_{2}, \ldots, x_{p}\right]\left[\begin{array}{c}\theta_{0} \\ \theta_{1} \\ \vdots \\ \theta_{p}\end{array}\right]
$$
损失函数：最小均方误差MSE：
$$
J(\theta)=\frac{1}{2} \sum_{i=1}^{n}\left(x_{i}^{T} \theta-y_{i}\right)^{2}
$$
线性回归模型：求解损失函数的最小值
$$
\theta^* = arg minJ(\theta)
$$

### 加入数据后的模型

n组数据

预测值：
$$
\hat Y = X\theta=\left[\begin{array}{l} X_1^T\theta \\X_2^T\theta \\ \ldots \\X_n^T\theta \\  \end{array}\right]=\left[\begin{array}{l} X_{11}\space X_{12}\ldots X_{1p}\\X_{21}\space X_{22}\ldots X_{2p} \\ \ldots \\X_{n1}\space X_{n2}\ldots X_{np} \\\end{array}\right]\left[\begin{array}{c}\theta_{0} \\ \theta_{1} \\ \vdots \\ \theta_{p}\end{array}\right]
$$
实际值label (n组数据n个label)：
$$
Y =\left[\begin{array}{c}y_1 \\ y_2\\ \vdots \\ y_n\end{array}\right]
$$



### 模型求解

#### 梯度下降法

Gradient Decent
$$
\theta:=\theta-\alpha \nabla_{\theta} J(\theta)
$$

$$
J(\theta)=\frac{1}{2} \sum_{i=1}^{n}\left(x_{i}^{T} \theta-y_{i}\right)^{2}
$$

其中算子：梯度是偏导数的自然扩展
$$
\nabla_{\theta} J=\left[\begin{array}{l}\frac{\partial J}{\partial \theta_{0}} \\ \cdots \\ \cdots  \\ \frac{\partial J}{\partial \theta_{p}}\end{array}\right]
$$
求损失函数的偏导：
$$
\begin{array}{l}\frac{\partial 1}{\theta_{j} 2}\left(x_{i}^{T} \theta-y_{i}\right)^{2} \\ =\frac{\partial 1}{\theta_{j} 2}\left(\sum_{j=0}^{p} x_{i, j} \theta_{j}-y_{i}\right)^{2} \quad x_{i}=\left(x_{i, 0}, \ldots, x_{i, p}\right)^{T} \\ =\left(\sum_{j=0}^{p} x_{i, j} \theta_{j}-y_{i}\right) \frac{\partial}{\theta_{j}}\left(\sum_{j=0}^{p} x_{i, j} \theta_{j}-y_{i}\right) \\ =\left(f\left(x_{i}\right)-y_{i}\right) x_{i, j}\end{array}
$$

#### 正规方程法

$$
\begin{aligned} J(\theta) &=\frac{1}{2}\|Y-X \theta\|^{2} \\ &=\frac{1}{2}(X \theta-Y)^{T}(X \theta-Y) \\ &=\frac{1}{2}\left(\theta^{T} X^{T} X \theta-2 Y^{T} X \theta+Y^{T} Y\right) \end{aligned}
$$

注解：
$$
\begin{array}{l}\frac{\partial \mathbf{x}^{T} \mathbf{B} \mathbf{x}}{\partial \mathbf{x}}=\left(\mathbf{B}+\mathbf{B}^{T}\right) \mathbf{x} \\ \frac{\partial \mathbf{x}^{T} \mathbf{a}}{\partial \mathbf{x}}=\frac{\partial \mathbf{a}^{T} \mathbf{x}}{\partial \mathrm{x}}=\text { a }\\\end{array}
$$
我们令$B=X^TX,B^T=B\Longrightarrow (B+B^B)\theta = 2B\theta$
$$
\nabla_{\theta} J(\theta)=\frac{\partial J(\theta)}{\partial \theta}=\frac{\frac{1}{2}\left(\theta^{T} X^{T} X \theta-2 Y^{T} X \theta+Y^{T} Y\right)}{\partial \theta}=X^{T} X \theta-\left(Y^{T} X\right)^{T}=X^{T} X \theta-X^{T} Y=0\\\Longrightarrow X^{T} X \theta=X^{T} Y\theta^{*}=\left(X^{T} X\right)^{-1} X^{T}\\\Longrightarrow\theta^{*}=\left(X^{T} X\right)^{-1} X^{T} Y
$$

#### 随机梯度下降法

Mini-batch GD

每次只 用训练集中的一个数据，把数据分为若干个批，按批来更新参 数。一个批中的一组数据共同决定了本次梯度的方向，下降起 来就不容易跑偏，减少了随机性。

一个bacth 形成一个epoch分批次训练

### 全局最优解

当$J(\theta)$是凸函数（凹函数和凸函数统称凸函数）时，二阶导数大于0,$X^TX$为半正定矩阵
$$
\nabla_{\theta}^{2} J(\theta)=X^{T} X
$$
当训练样本的数目n大于训练样本的维度（p+1 个属性，特征）$X^TX$通常可逆，表明改矩阵事正定矩阵，求的参数是全局最优解。不可逆时，可以接出多个参数解。可使用 正则化给出一个“归纳偏好”解。

### 评估方法

#### 留出法

随机挑选 一部分标 记数据作 为测试集 (空心点 )，其余的作 为训练集 (实心点 )，计算 回归模型，使用测试 集对模型 评估: MSE =2.4，测试集不能太大，也不 能太小。2 <= n:m <=4

#### 交叉验证法

![](https://cdn.mathpix.com/snip/images/nXRmmZcFN_wIuR7Nc-faI45CWKH5hS6nU-eZ3hlYD70.original.fullsize.png)

#### 性能度量

##### 线性回归模型：平方和误差

在测试集上报告 MSE(mean square error) 误差
$$
J_{\text {train }}(\theta)=\frac{1}{2} \sum_{i=1}^{n}\left(\mathbf{x}_{i}^{T} \theta-y_{i}\right)^{2}
$$

$$
\theta^{*}=\operatorname{argmin} J_{\text {train }}(\theta)=\left(X_{\text {train }}^{T} X_{\text {train }}\right)^{-1} X_{\text {train }}^{T} \vec{y}_{\text {train }}
$$

$$
J_{\text {test }}=\frac{1}{m} \sum_{i=n+1}^{n+m}\left(\mathbf{x}_{i}^{T} \theta^{*}-y_{i}\right)^{2}=\frac{1}{m} \sum_{i=n+1}^{n+m} \varepsilon_{i}^{2}
$$

##### 分类任务：错误率与精度

错误率是分类错误的样本数占样本总数的比例

精度是分类正确的样本数占样本总数的比例

对二分类问题：

查准率：$P=\frac{T P}{T P+F P}$

查全率：$R=\frac{T P}{T P+F N}$

F1:
$$
F 1=\frac{2 \times P \times R}{P+R}=\frac{2 \times T P}{\text { 样例总数 }+T P-T N}
$$



## 基于非线形基的线性回归

### 多项式回归



# LR-逻辑回归 

## Structural model
逻辑函数（logistic/sigmoid function）

$$
y=\frac{1}{1+e^{-z}} = \frac{1}{1+e^{-\theta x}}
$$

## Error model
损失函数 Loss function 

$$
\begin{array}{c}P(y=1 \mid x ; \theta)=f_{\theta}(x)=\frac{1}{1+e^{-\theta^{T} * x}} \\ P(y=0 \mid x ; \theta)=1-f_{\theta}(x)=\frac{e^{-\theta^{T} * x}}{1+e^{-\theta^{T} * x}}\end{array}
$$


求参数方法--对$\theta$极大似然估计，使得y发生的概率最大
$$
L(\theta)=\prod_{i=1}^{n} P\left(y_{i} \mid x_{i} ; \theta\right)=\prod_{i=1}^{n}\left(f_{\theta}\left(x_{i}\right)\right)^{y_{i}}\left(1-f_{\theta}\left(x_{i}\right)\right)^{1-y_{i}}
$$
转化为对数函数，将$\frac{f_{\theta}(x)} {1-f_{\theta}(x)}=\frac{1}{1+e^{-\theta^{T} * x}}/\frac{e^{-\theta^{T} * x}}{1+e^{-\theta^{T} * x}} = \frac{1}{e^{-\theta x}} = e^{\theta x}$
$$
\begin{array}{l}\ln L(\theta)=\sum_{i=1}^{n}\left(y_{i} \ln \left(f_{\theta}\left(x_{i}\right)\right)+\left(1-y_{i}\right) \ln \left(1-f_{\theta}\left(x_{i}\right)\right)\right) \\ =\sum_{i=1}^{n}\left(\left(1-y_{i}\right)\left(-\theta^{T} * x_{i}\right)-\ln \left(1+e^{-\theta^{T} * x_{i}}\right)\right)\end{array}
$$
梯度上升
$$
\theta:=\theta+\alpha \nabla_{\theta} \ln (L(\theta)) \Leftrightarrow \theta_{j}:=\theta_{j}+\frac{\partial \ln (L(\theta))}{\partial \theta_{j}}
$$
求梯度
$$
\begin{aligned} \nabla_{\theta} \ln (L(\theta)) &=\sum_{i=1}^{n}\left[-\left(1-y_{i}\right) \cdot x_{i}-\frac{1}{1+e^{-\theta^{T} x_{i}}}\left(e^{-\theta^{T} x_{i}}\right)\left(-x_{i}\right)\right] \\ &=\sum_{i=1}^{n}\left(-1+y_{i}+\frac{e^{-\theta^{T} x_{i}}}{1+e^{-\theta^{T} x_{i}}}\right) x_{i} \\ &=\sum_{i=1}^{n}\left(y_{i}-f_{\theta}\left(x_{i}\right)\right) x_{i} \Leftrightarrow \frac{\partial}{\partial \theta_{j}} \ln (L(\theta))=\sum_{i=1}^{n}\left(y_{i}-f_{\theta}\left(x_{i}\right)\right) x_{i, j} \end{aligned}
$$
代入梯度的参数更新
$$
\theta:=\theta+\alpha \nabla_{\theta} \ln (L(\theta)) \Rightarrow \theta:=\theta+\alpha \sum_{i=1}^{n}\left(y_{i}-f_{\theta}\left(x_{i}\right)\right) x_{i}
$$
## 和线性回归的对比

和线形回归模型看似一样，但是f不同，逻辑回归解决的是二分类问题

|      | 逻辑回归   | 线性回归 |
| ---- | ---------- | -------- |
| 输出 |            |          |
|      | 线形二分类 | 线性拟合 |
|      |            |          |
|      |            |          |
|      |            |          |

# NN-神经网络
## Structural model

### 逻辑回归的二阶段表示

$z = b+ \sum x_iw = \mathop{W^T}\limits_{p \times 1}\mathop{x}\limits_{1\times p} + \mathop{b}\limits_{1\times1}$

$\hat y = sigmoid(z) = \frac{e^z}{1+e^z}$

### 神经元

神经元=线性组合(z，接收信号)+非线性激活(sigmoid， 输出非线性决策面)
$$
\boldsymbol{z}_{t}=W_{1}^{T} \boldsymbol{x}
$$
多神经元

神经网络包含多个神经元， 输入x与多个神经元相连。

### 一个隐藏层的神经网络

$$
\boldsymbol{z}_{1}=W_{1}^{T} \boldsymbol{x}\\h_1 = sigmoid(z_1)\\z_2 = w_2^Th_1\\ \hat y = sigmoid(z_1)
$$
W表示X的第j个元素与向量Z的第i个元素之间的链接权重
$$
W=\left[\begin{array}{llll}W_{11} & W_{21} & W_{31} & W_{41} \\ W_{12} & W_{22} & W_{32} & W_{42} \\ W_{13} & W_{23} & W_{33} & W_{43}\end{array}\right]
$$

$$
\mathrm{W}^{T}=\left[\begin{array}{lll}W_{11} & W_{12} & W_{13} \\ W_{21} & W_{22} & W_{23} \\ W_{31} & W_{32} & W_{33} \\ W_{41} & W_{42} & W_{43}\end{array}\right]
$$

隐含层h
没有隐含层就只需要一个列向量，因为有隐含层所以需要W矩阵
每一层计算就是线性组合+非线形激活

### 非线形激活函数

引入非线性激活函数的目的是得到非线性决策面，非线形激活函数可以逼近任何复杂的函数，不论网络多深，线形函数只能输出线性决策面。

非线形激活函数
Relu效果最好，因为有部分导数为0，有些为1，为0的部分可以让有些神经元停止学习，起到dropout的作用，可以有效防止过拟合。

binary step
$$
f(x)=\left\{\begin{array}{lll}0 & \text { for } & x<0 \\ 1 & \text { for } & x \geq 0\end{array}\right.
$$
Logistic
$$
f(x)=\frac{1}{1+e^{-x}}
$$
Tanh
$$
f(x)=\tanh (x)=\frac{2}{1+e^{-2 x}}-1
$$
ReLU
$$
f(x)=\left\{\begin{array}{lll}0 & \text { for } & x<0 \\ x & \text { for } & x \geq 0\end{array}\right.
$$

### 多分类神经网络

<img src="/Users/wangjing/Library/Application Support/typora-user-images/image-20211019140804340.png" alt="image-20211019140804340" style="zoom:45%;" />
$$
\begin{array}{l}\boldsymbol{z}_{1}=\boldsymbol{W}_{1}^{T} \boldsymbol{x} \\ h_{1}=\operatorname{sigmoid}\left(z_{1}\right) \\ z_{2}=\boldsymbol{W}_{2}^{T} h_{1} \\ h_{2}=\operatorname{sigmoid}\left(z_{2}\right) \\ \boldsymbol{z}_{3}=w_{3}^{T} h_{2} \\ \hat{y}=\operatorname{sigmoid}\left(z_{3}\right)\end{array}
$$
$h_1$表示hidden layer 1 output

Hidden layer(隐层)的个数大于1的神经网络，称为深度神经网络

## Error model

非正确预测导致的代价

### Loss function

交叉熵函数（cross entropy loss）

#### 二分类损失

逻辑回归中，使用对数似然度量损失(每个样本属于其真实 标记的概率越大越好)
$$
\begin{aligned} E=\operatorname{loss} &=-\log P(\mathrm{Y}=\hat{y} \mid \mathbf{X}=\boldsymbol{x}) \\ &=-y \log (\hat{y})-(1-y) \log (1-\hat{y}) \end{aligned}
$$

#### 多分类损失

##### Softmax函数

(柔性 最大值):将输出值转化成概率。
$$
\hat{y}_{i}=\frac{e^{z_{i}}}{\sum_{j=1}^{K} e^{z_{j}}}=\mathrm{P}\left(y_{i}=1 \mid \mathrm{x}\right)
$$
$y_j$为one-hot 向量,真实标签位置1其他位置为0
$$
E=\operatorname{loss}=-\sum_{j=1 . K} y_{j} \log \hat{y}_{j}
$$

#### 回归损失

与分类网络不同：输出层（最后一层）不再包含sigmoid函数

##### 二次代价函数

$$
\begin{array}{l}E=\operatorname{Los} s=\frac{1}{2}\|y-\hat{y}\|^{2} \\ =\frac{1}{2} \sum_{j=1}^{K}\left(y_{j}-\hat{y}_{j}\right)^{2}\end{array}
$$

## 模型建模

优化参数目标:寻找使损失达到最小的神经网络权重
$$
\mathrm{W}^{*}=\underset{W}{\operatorname{argmin}} E(\hat{y} ; \mathrm{W})
$$
如何学习实现目标的神经网络权重𝑊 --梯度下降
$$
W_{L}(t+1)=W_{L}(t)-\eta \frac{\partial E}{\partial W_{L}(t)}
$$
### 反向传播

求偏导从而应用梯度下降

1. 重复应用微积分的链式法则
2. 局部最小化目标函数
3. 要求网络所有的“块”(blocks)都是可微的

```
正向计算--节点
反向求导--边 链式法则从后往前求
```

#### 反向传播--回归实例

回归损失函数为二次代价函数
$$
E = loss = \frac{1}{2}(y-\hat y)^2
$$

#### 反向传播--二分类实例

二分类损失函数为交叉熵损失函数
$$
\text { Loss }=-y \ln (\widehat{y})-(1-y) \ln (1-\widehat{y})
$$
通过梯度下降 最小化Loss
$$
\begin{array}{l}w_{2}(t+1)=w_{2}(t)-\eta \frac{\partial E}{\partial w_{2}(t)} \\ W_{1}(t+1)=W_{1}(t)-\eta \frac{\partial E}{\partial W_{1}(t)}\end{array}
$$

函数关于一个矩阵求偏导-->对每一个元素求偏导,$W_{11}^1$表示输入x的第j个元素到第一个隐层的第i个神经元的权重
$$
\begin{aligned} E=&-y \ln (\hat{y}) -(1-y) \ln (1-\hat{y}) \\ \hat{y}=& \frac{e^{z_{2}}}{1+e^{z_{2}}} \\ z_{2}=& \boldsymbol{w}_{2}^{T} \boldsymbol{h}_{1} \\ \boldsymbol{h}_{1}=& \frac{e^{z_{1}}}{1+e^{z_{1}}} \\ \boldsymbol{z}_{1}=& W_{1}^{T} \boldsymbol{x} \end{aligned}
$$

$$
\frac{\partial E}{\partial \boldsymbol{W}_{1}}=\frac{\partial E}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_{2}} \cdot \mid \frac{\partial z_{2}}{\partial \boldsymbol{h}_{1}} \cdot \frac{\partial \boldsymbol{h}_{1}}{\partial \boldsymbol{z}_{1}} \cdot \frac{\partial \boldsymbol{z}_{1}}{\partial W_{1}}
$$

**Hadamard (哈达玛)乘积 /schur 乘积**
假设 𝑠和𝑡是两个同样维度的向量，使用𝑠 ∘ 𝑡(或𝑠 ⊙ 𝑡)来表示按元素的乘积: $(𝑠⊙𝑡) =s_jt_j$
$$
\left[\begin{array}{l}1 \\ 2\end{array}\right] \odot\left[\begin{array}{l}3 \\ 4\end{array}\right]=\left[\begin{array}{l}1 * 3 \\ 2 * 4\end{array}\right]=\left[\begin{array}{l}3 \\ 8\end{array}\right]
$$

**反向传播的局部性**

反向传播的一般情形
第𝑙层第𝑗个神经元和第𝑙 − 1 层神经元之间关系
$$
z_{j}^{l}=\sum_{k=1} w_{j k}^{l-1} h_{k}^{l-1}+b_{j}^{l-1}
$$

#### 反向传播的一般情形

一些定义

$\delta_{j}^{l}: \quad \delta_{j}^{l} \equiv \frac{\partial E}{\partial z_{j}^{l}}$，称为在第𝑙层第𝑗个神经元的误差
$$
\begin{array}{l}z_{j}^{l}=\sum_{k=1} w_{j k}^{l-1} h_{k}^{l-1}+b_{j}^{l-1} \\ h_{j}^{l}=\sigma\left(z_{j}^{l}\right) \\ \sigma(x)=\frac{1}{1+e^{-x}}\end{array}
$$
矩阵表达形式--代价函数

$$
E=\frac{1}{2}\left\|\boldsymbol{y}-\boldsymbol{h}^{L}\right\|^{2}=\frac{1}{2}\|\boldsymbol{y}-\widehat{\boldsymbol{y}}\|^{2}
$$
第𝑙层第𝑗个神经元和第𝑙 − 1层神经元之间的关系:
$$
z_{j}^{l}=\sum_{k=1} w_{j k}^{l-1} h_{k}^{l-1}+b_{j}^{l-1}, \quad h_{j}^{l}=\sigma\left(z_{j}^{l}\right)
$$



### 🐮反向传播四个方程

#### BP1

输出层（最后一层，即为L层）误差的方程
$$
E=\frac{1}{2}\left\|\boldsymbol{y}-\boldsymbol{h}^{L}\right\|^{2}=\frac{1}{2} \sum_{j=1}^{K}\left(h_{j}^{L}-y_{j}\right)^{2}
$$
第L层第j个神经元的误差
$$
\delta_{j}^{L}=\frac{\partial E}{\partial z_{j}^{L}}=\frac{\partial E}{\partial h_{j}^{L}} \frac{\partial h_{j}^{L}}{\partial z_{j}^{L}}=\left(h_{j}^{L}-y_{j}\right) \sigma^{\prime}\left(z_{j}^{L}\right)
$$
向量表达形式：
$$
𝛿_𝐿=(𝒉^𝐿−𝑦)\odot𝜎^{'}(𝒛^𝐿)
$$

#### BP2

每一层的误差，使用下一层的误差 $\delta^{l+1} $表示当前层的误差 $\delta^l $:
$$
\delta^{l}=\sigma^{\prime}\left(\mathbf{z}^{l}\right) \odot\left(\boldsymbol{W}^{l} \delta^{l+1}\right)
$$



#### BP3

代价函数关于偏置b的偏导

#### BP4

代价函数关于权重的偏导

#### Summary

向量形式：
$$
\begin{array}{l}\text { (BP1) } \delta^{L}=\left(\boldsymbol{h}^{L}-y\right) \odot \sigma^{\prime}\left(\mathbf{z}^{L}\right)\\ \text { (BP2) } \delta^{l}=\sigma^{\prime}\left(\mathbf{z}^{l}\right) \odot\left(\boldsymbol{W}^{l} \delta^{l+1}\right) \\ \text { (BP3) } \frac{\partial E}{\partial b^{l-1}}=\delta^{l} \\ \text { (BP4) } \frac{\partial E}{\partial W^{l-1}}=\boldsymbol{h}^{l-1}\left(\delta^{l}\right)^{T}\end{array}
$$

数学形式：
$$
BP1:& \delta_{j}^{L}=\left(h_{j}^{L}-y_{j}\right) \sigma^{\prime}\left(z_{j}^{L}\right)\\BP2:&\delta_{j}^{l}=\sum_{k=1} \delta_{k}^{l+1} w_{k j}^{l} \sigma^{\prime}\left(z_{j}^{l}\right)\\BP3:&\frac{\partial E}{\partial b_{j}^{l-1}}=\delta_{j}^{l}\\BP4:&\frac{\partial E}{\partial w_{j k}^{l-1}}=h_{k}^{l-1} \delta_{j}^{l}
$$

反向传播算法
1. 输入x：为输入层设置对应的激活值h1
2. 前向传播：线性组合+非线形激活
3. 输出层误差和反向误差传播：BP1和BP2
4. 输出：误差函数的梯度由BP3和BP4给出


## 模型改进

### 改进损失函数

对数似然

#### 损失函数对比
交叉熵VS二次代价函数

|               | 二次代价函数                                                 | 交叉熵                                                       |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 函数表达式    | $E = \frac{1}{2}(y-\hat y)^2$                                | $E = -yln\hat y-(1-y)ln(1-\hat y)$                           |
|               |                                                              |                                                              |
|               |                                                              |                                                              |
| 对参数w偏导   | $\frac{\partial E}{\partial w}=(\hat{y}-y) \sigma^{\prime}(z) x=(\sigma(z)-y) \sigma^{\prime}(z) x$ | $\frac{\partial E}{\partial w}=(\hat{y}-y) x=(\sigma(z)-y) x$ |
| 对参数b的求导 | $\frac{\partial E}{\partial b}=(\hat{y}-y) \sigma^{\prime}(z)=(\sigma(z)-y) \sigma^{\prime}(z)$ | $\frac{\partial E}{\partial b}=(\hat{y}-y)=(\sigma(z)-y)$    |

### 权重初始化建议

随机初始化:使用Numpy的 np.random.randn函数生成均 值为0，标准差为1的高斯分布。

改进:对于任意𝑙层，使用均值为0，标准差为1 的高斯分布随机分布初始化权重参数𝑊𝑙−1，𝑏𝑙−1。此时中间变量𝑧𝑙 服从均值为0，标准差为1的高斯分布。

###  减少过拟合:dropout 

1. 随机地删除网络中的一半的隐藏神经元，同时让输入层和输出层的神经元保持不变。
2. 把输入x通过修改后的网络前向传播，然后把得到的损失结 果通过修改的网络反向传播。在mini-batch 上执行完这个过 程后，在没有被删除的神经元上更新对应的参数(w，b)
3. 继续重复上述过程:
	- 恢复被删掉的神经元(此时被删除的神经元保持原样， 而没有被删除的神经元已经有所更新)
	- 从隐藏层神经元中随机选择一个一半大小的子集临时 删除掉(备份被删除神经元的参数)。
	- 对一小批训练样本，先前向传播然后反向传播损失并 更新参数(w，b) (没有被删除的那一部分参数得到 更新，删除的神经元参数保持被删除前的结果)。

### 缓解梯度消失:ReLU

当𝑧 是负数的时候，梯度为0，神经元停止学习(类似于 dropout作用，减少过拟合);当𝑧大于0时，梯度为1，可 以缓解下溢问题



# SVM-支持向量机

## 线形可分-SVM

约束优化问题

1. 目标函数 $minf(x)$

2. 变量

3. 约束条件
   $$
   \begin{array}{lllll}\text { s.t. } & g_{j}(x)=0, & j=1, & 2, & \cdots & n \\ & h_{i}(x) \leq 0, & i=1, & 2, & \cdots, & m\end{array}
   $$

求解方法---拉格朗日乘数法
$$
L(x, \lambda, \alpha)=f(x)+\sum_{j} a_{j} g_{j}(x)+\sum_{i} \lambda_{i} h_{i}(x)
$$

$$
\frac{\partial L}{\partial x} = \nabla f\left(x^{*}\right)+\sum_{i} a_{j} \nabla g_{j}\left(x^{*}\right)+\sum_{i} \lambda_{i} \nabla h_{i}\left(x^{*}\right)=0
$$

$$
\begin{array}{l}\nabla_{\mathbf{x}} L=\frac{\partial L}{\partial \mathbf{x}}=\nabla f+\lambda \nabla g=\mathbf{0} \\ \nabla_{\lambda} L=\frac{\partial L}{\partial \lambda}=g(\mathbf{x})=0\end{array}
$$
计算 L 对 x 与 $\lambda$ 的偏导数并设为零，可得最优解的必要条件
如何理解KKT？

Karush-Kuhn-Tucker (KKT)条件

非线性规划最佳解的必要条件--KKT条件将Lagrange乘数法所处理涉及等式的约束优化问题推广至不等式
$$
\begin{array}{c}\nabla f\left(x^{*}\right)+\sum_{j} a_{j} \nabla g_{j}\left(x^{*}\right)+\sum_{i} \lambda_{i} \nabla h_{i}\left(x^{*}\right)=0 \\ g_{j}\left(x^{*}\right)=0 \\ h_{i}\left(x^{*}\right) \leq 0, \lambda_{i} \geq 0, \lambda_{i} h_{i}\left(x^{*}\right)=0\end{array}
$$
### 对偶问题
$$
\max _{\alpha_{i} \geq 0} \min _{w} L(w, \alpha)
$$
$$
f_{0}(w) = \max _{\alpha_{i} \geq 0} L(w, \alpha)\\
f_{0}(w) > L(w, \alpha)
$$

简单的例子
$$
\begin{array}{l}\min _{u} u^{2} \\ \text { s.t. } u>=b\end{array}
$$
使用拉格朗日乘数法将其转化为
$$
L = u^2 + \lambda (u-b) \\
\frac{\partial L}{\partial u} = 2u + \lambda
\\ u-b = 0 
\\u = b
\\\lambda = 2b
$$

### Margin模型

<img src="/Users/wangjing/Library/Application Support/typora-user-images/image-20211116103423027.png" alt="image-20211116103423027" style="zoom:50%;" />

分类面：
$$
w^{T} x+b=0
$$
+1支持面：
$$
w^{T} x+b = 1
$$
-1支持面：
$$
w^{T} x+b=-1
$$
向量 w 与支持面、分类面正交
$$
\left.\begin{array}{l}w^{T} x_{1}+b=1 \\ w^{T} x_{2}+b=1\end{array}\right\} \Rightarrow w^{T}\left(x_{1}-x_{2}\right)=0
$$
使用 w 和 b 对 M 建模
$$
\left.\begin{array}{l}w^{T} x^{+}+b=+1 \\ w^{T} x^{-}+b=-1\end{array}\right\} \Rightarrow w^{T}\left(x^{+}-x^{-}\right)=2
$$
得到两个支撑面最大间隔
$$
\operatorname{margin} M=\left\|x^{+}-x^{-}\right\|=\frac{2}{\|w\|}
$$

### 分类模型

目标函数：间隔最大 （二次函数）


$$
\max \left(\frac{2}{\|w\|}\right) \Leftrightarrow \min \left(\|w\|^{2}\right) \\ \min _{\boldsymbol{w}, b} \boldsymbol{w}^{T} \boldsymbol{w} / 2
$$
约束:线形约束
$$
\left\{\begin{array}{ll}\boldsymbol{w}^{T} \boldsymbol{x}_{i}+b \geq 1 & y_{i}=1 \\ \boldsymbol{w}^{T} \boldsymbol{x}_{i}+b \leq-1 & y_{i}=-1\end{array}\right.
$$
约束可合并为：
$$
y_{i}\left(\boldsymbol{w}^{T} x_{i}+b\right) \geq 1
$$

## SVM--进阶

### 线形不可分SVM-软-SVM

新的优化问题
$$
\min _{w, b} w^{T} w / 2+C \sum_{i=1}^{n} \epsilon_{i}
$$
约束：
$$
\begin{array}{c}y_{i}\left(w^{T} x_{i}+b\right) \geq 1-\epsilon_{i} \\ \epsilon_{i} \geq 0\end{array}
$$
软-SVM对偶问题
$$
\begin{array}{l}\max _{\alpha} \sum_{i} \alpha_{i}-\frac{1}{2} \sum_{i, j} \alpha_{i} \alpha_{j} y_{i} y_{j} \mathbf{x}_{i}^{T} \mathbf{x}_{j} \\ \sum_{i} \alpha_{i} y_{i}=0 \\ C \geq \alpha_{i} \geq 0, \forall i\end{array}
$$
用拉格朗日乘数法转华为无约束问题：
$$
L=\frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{n} \epsilon_{i}-\sum_{i=1}^{n} \alpha_{i}\left(y_{i}\left(w^{T} x_{i}+b\right)-1+\epsilon_{i}\right)-\sum_{i=1}^{n} \mu_{i} \epsilon_{i}
$$

$$
\frac{\partial L}{\partial w} = 0 \\
\frac{\partial L}{\partial \alpha} = 0 \\
\frac{\partial L}{\partial \mu} = 0
$$

支持向量有两类：

1. 支持面上的点
2. 违背硬约束样本点

### 非线形-核SVM

模型：

1. 利用非线性映射把原始数据映射到高维空间中$\phi(x)$
2. 目标函数

$$
\begin{array}{l}\min w^{T} w / 2 \\ \text { s.t. } y_{i}\left(w^{T} \phi\left(x_{i}\right)+b\right) \geq 1\end{array}
$$

### 多分类SVM

1. 一对多one-verus-rest

一种正样本，多种负样本

会出现数据不平衡，分类面偏置
$$
\hat y \leftarrow argmaxw_k x + b_k
$$
改进：期望正类和负类之间的错误达到平衡
$$
\begin{array}{c}\min w^{T} w / 2+\mathrm{C}\left(\frac{N}{N_{+}} \sum_{i: y_{i}=+1} \epsilon_{i}+\frac{N}{N_{-}} \sum_{i: y_{i}=-1} \epsilon_{i}\right) N=N_{+}+N_{-} \\ \text {s.t. } \quad y_{i}\left(w^{T} x_{i}+b\right) \geq 1-\epsilon_{i} \\ \epsilon_{i} \geq 0\end{array}
$$

2. 多个1V1 one-verus-one 训练 $\frac{m(m-1)}{2}$个分类器

样本量较少，分类器数量更多，测试成本高



## 🍑SVR 支持向量回归

结论：

利用KKT条件：
$$
\left\{\begin{array}{c}\alpha_{i}\left(y_{i}-f\left(x_{i}\right)-\epsilon-\xi_{i}\right)=0 \\ \hat{\alpha}_{i}\left(f\left(x_{i}\right)-y_{i}-\epsilon-\hat{\xi}_{i}\right)=0 \\ \xi_{i}\left(C-\alpha_{i}\right)=0 \\ \hat{\xi}_{i}\left(C-\hat{\alpha}_{i}\right)=0\end{array}\right.
$$

1. 当0 < 𝛼𝑖 < 𝐶, 𝑥𝑖落在间隔带上边界
2. 当𝛼𝑖 = 𝐶 时，𝑥𝑖 落在间隔带上边界外侧
3. 当0 < 𝛼𝑖 < 𝐶，𝑥𝑖落在间隔带下边界
4. 当𝛼𝑖 = 𝐶 时，𝑥𝑖 落在间隔带下边界外侧
5. 当𝛼𝑖 = 𝛼𝑖 = 0时，点落在间隔带内侧



# 特征选择和稀疏学习

## 特征选择
## 过滤式选择
(Filter method)
单变量(Univariate)过滤方法:Signal-to-noise ratio (S2N)
$$
\mathrm{S} 2 \mathrm{~N}=\frac{|\mu+-\mu-|}{\sigma^{+}+\sigma-}
$$
多变量(Multivariate)过滤方法:Relief

给定训练集 𝑥 ,𝑦 ,..., 𝑥 ,𝑦 ,

1、对每个样本 𝑥𝑖，在同类样本中找最近邻 𝑥𝑖,h𝑖𝑡;在异类样本中寻找最近邻 𝑥𝑖,𝑚𝑖𝑠𝑠

2、计算对应于属性 𝑗 的统计量
$$
\delta^{j}=\sum_{i}-\operatorname{diff}\left(x_{i}^{j}, x_{i, h i t}^{j}\right)^{2}+\operatorname{diff}\left(x_{i}^{j}, x_{i, m i s s}^{j}\right)^{2}
$$
3、若𝛿𝑗大于指定阈值𝜏，选择属性𝑗;或者指定一个k值，选择统计量最大的前k 个特征



## 包裹式选择

(Wrapper method) 

将所有属性作为一个集合，每次从中选出部分作为训练特征。

NP难问题

寻找最优子集

验证集：选超参



## 🍑嵌入式选择--正则化

(Embedded method)

L1正则化
$$
E=\frac{1}{2 n} \sum_{x}\left\|\boldsymbol{y}^{x}-\boldsymbol{h}^{x, L}\right\|^{2}+\frac{\eta}{2 n} \sum_{l}\left\|w^{l}\right\|_{1}
$$
L2正则化
$$
E=\frac{1}{2 n} \sum_{x}\left\|y^{x}-h^{x, L}\right\|^{2}+\frac{\eta}{2 n} \sum_{l}\left\|w^{l}\right\|_{2}^{2}
$$
混合正则化
$$
E=\frac{1}{2 n} \sum_{x}\left\|\boldsymbol{y}^{x}-\boldsymbol{h}^{x, L}\right\|^{2}+\frac{\beta}{2 n} \sum_{l}\left\|W^{l}\right\|_{1}+\frac{\eta}{2 n} \sum_{l}\left\|W^{l}\right\|_{2}^{2}
$$

对偏置b不进行正则化，只对权重w进行正则化，假设$\sum_{i=1}^{n} x_{i}=0$,
$$
\beta_{0}=\frac{1}{n} \sum_{i=1}^{n} y_{i}
$$

$$
\min _{\beta, \beta_{0}} \sum_{i=1}^{n}\left(y_{i}-\boldsymbol{\beta}^{T} x_{i}-\beta_{0}\right)^{2}+\lambda \sum_{j=1}^{p}\left|\beta_{j}\right|^{q}
$$
### L2-正则化

q=2，，使用L2范数正则化称为ridge regression，岭回归
$$
\min _{\boldsymbol{\beta}, \boldsymbol{\beta}_{0}} \sum_{i=1}^{n}\left(y_{i}-\boldsymbol{\beta}^{T} \boldsymbol{x}_{\boldsymbol{i}}-\beta_{0}\right)^{2}+\lambda \sum_{j=1}^{p}\left|\beta_{j}\right|^{q}化为有约束形式：（why ？ 为什么要这样化）
$$

$$
\begin{array}{l}\min _{\boldsymbol{\beta}, \beta_{0}} \sum_{i=1}^{n}\left(y_{i}-\boldsymbol{\beta}^{T} \boldsymbol{x}_{i}-\beta_{0}\right)^{2} \\ \text { s. t. }\|\boldsymbol{\beta}\|^{2} \leq t\end{array}
$$
求解方法：化为矩阵形式，利用正规方程法对其进行求解
$$
L = \|Y-X \beta\|^{2} +\lambda||\beta||^2
$$
对权重参数求偏导，二范数的矩阵表示
$$
\|x\|_{2}=\sqrt{\sum_{i=1}^{n} x_{i}^{2}}=\sqrt{\mathbf{X}^T \mathbf{X}}
$$

$$
\frac{\partial L}{\partial \beta}=2 X^T(X \beta-Y)+2 \lambda \beta=0
$$

$$
\Rightarrow X^T X \beta-X^T Y+\lambda \beta=0
$$

$$
\Rightarrow \left(X^TX+\lambda\right) \beta=X^T Y
$$
求解得到：
$$
\Rightarrow \beta=\left(X^TX+\lambda I\right)^{-1} X^T Y
$$
#### （*）SVD 奇异值分解

用SVD解释岭回归: $𝑋=𝑈𝐷𝑉^𝑇$，𝑈为𝑛×𝑝,𝑉为𝑝×𝑝正交矩 阵，𝐷为对角阵，满足𝑑1 ≥𝑑2 ≥⋯≥𝑑𝑝 ≥0
$$
\begin{aligned} \mathbf{X} \hat{\beta}^{\mathrm{ls}} &=\mathbf{X}\left(\mathbf{X}^{T} \mathbf{X}\right)^{-1} \mathbf{X}^{T} \mathbf{y} \\ &=\mathbf{U} \mathbf{U}^{T} \mathbf{y} \\ \mathbf{X} \hat{\beta}^{\text {ridge }} &=\mathbf{X}\left(\mathbf{X}^{T} \mathbf{X}+\lambda \mathbf{I}\right)^{-1} \mathbf{X}^{T} \mathbf{y} \\ &=\mathbf{U} \mathbf{D}\left(\mathbf{D}^{2}+\lambda \mathbf{I}\right)^{-1} \mathbf{D} \mathbf{U}^{T} \mathbf{y} \\ &=\sum_{j=1}^{p} \mathbf{u}_{j} \frac{d_{j}^{2}}{d_{j}^{2}+\lambda} \mathbf{u}_{j}^{T} \mathbf{y} \end{aligned}
$$
### L1-正则化

Least Absolute Shrinkage and Selection Operator, Lasso回归

q=1，w变成0，自动放弃特征，起到特征选择，防止过拟合的方法，可以使得特征矩阵稀疏。
$$
\begin{array}{l}\min _{\boldsymbol{\beta}, \boldsymbol{\beta}_{0}} \sum_{i=1}^{n}\left(y_{i}-\boldsymbol{\beta}^{T} \boldsymbol{x}_{\boldsymbol{i}}-\beta_{0}\right)^{2} \\ \text { s.t. }\|\boldsymbol{\beta}\|_{1} \leq t\end{array}
$$
等价拉格朗日表达形式，用拉格朗日乘数法，化有条件极值为无条件极值：
$$
\min _{\boldsymbol{\beta}, \boldsymbol{\beta}_{0}} \sum_{i=1}^{n}\left(y_{i}-\boldsymbol{\beta}^{T} \boldsymbol{x}_{\boldsymbol{i}}-\beta_{0}\right)^{2}+\lambda\|\boldsymbol{\beta}\|_{1}
$$
L1约束使得解关于 𝒚 非线性，而且不能像岭回归那样可以得 到封闭解。

1. 闭式解需要满足正交性$X^TX=I$
2. 一般方法Lasso回归求解(一般情形): 坐标下降法(CoordinateDescent)
    相当于每次迭代都只是更新x的一个 维度，即把该维度当做变量，剩下 的n-1个维度当作常量,通过最小化f(x) 来找到该维度对应的新的值。
3. 坐标 下降法就是通过迭代地构造序列 $x^{0}, x^{1}, x^{2},...$来求解问题，即 最终点收敛到期望的局部极小值点

  

### Lasso回归
$$
\min _{\boldsymbol{\beta}, \beta_{0}} \sum_{i=1}^{n}\left(y_{i}-\boldsymbol{\beta}^{T} \boldsymbol{x}_{i}-\beta_{0}\right)^{2}+\lambda\|\boldsymbol{\beta}\|_{1}
$$
每次针对一个属性进行更新，为什么$\beta_0$下面的推到没了
$$
\mathrm{L}=\sum_{i=1}^{n}\left(y_{i}-\sum_{j=1}^{p} x_{i, j} \beta_{j}\right)^{2}+\lambda \sum_{j=1}^{p}\left|\beta_{j}\right|
$$
完全平方式展开，L对权重参数求导
$$
\frac{\partial L}{\partial \beta_{k}}=2 a_{k}+2 b_{k} \beta_{k}+\lambda \operatorname{sign}\left(\beta_{k}\right), \\\text { where } a_{k}=\sum_{i=1}^{n} x_{i, k}\left(\sum_{j \neq k}^{p} x_{i, j} \beta_{j}-y_{i}\right), b_{k}=\sum_{i=1}^{n} x_{i, k}^{2}
$$
利用$\frac{\partial L}{\partial \beta_{k}}=0 $,得到
$$
\beta_{k}=\left\{\begin{array}{l}-\frac{1}{b_{k}}\left(a_{k}-\frac{\lambda}{2}\right), \quad a_{k}>\frac{\lambda}{2} \\ 0, \quad-\frac{\lambda}{2}<a_{k}<\frac{\lambda}{2} \\ -\frac{1}{b_{k}}\left(a_{k}+\frac{\lambda}{2}\right), a_{k}<-\frac{\lambda}{2}\end{array}\right.
$$



## 稀疏表示字典学习

字典学习：给定数据集${x_1, x_2, ..., x_n}$
$$
\min _{B, \alpha_{i}} \sum_{i=1}^{n}\left(\left\|x_{i}-B \alpha_{i}\right\|^{2}+\lambda\left\|\alpha_{i}\right\|_{1}\right)
$$

- 其中𝑩 ∈ 𝑅𝑝×𝑘 为字典矩阵， 𝑘为字典的词汇量(通常由用户指定)， 𝜶𝒊 ∈ 𝑅𝑘是样本𝒙𝑖 ∈ 𝑅𝑝 的稀疏表示。

求解方法：交替优化（控制变量法）

1. 固定B，优化$\alpha$---Lasso回归问题-------坐标下降法求解
2. 固定$\alpha$,优化B-----线形优化-------正规方程法求解



# 集成学习

思想：民主决策，少数服从多数

好的集成：个体要有差异，个体精度不能太低：好而不同



集成的有效性：
$$
H(\boldsymbol{x})={\operatorname{sign}}\left(\sum_{i=1}^{T} h_{i}(\boldsymbol{x})\right)
$$

分类错误率随着T的增大呈指数下降
$$
\begin{aligned} P(\stackrel{H(\boldsymbol{x})} \neq {f(\boldsymbol{x})}) &=\sum_{k=0}^{\lfloor T / 2\rfloor}\left(\begin{array}{c}T \\ k\end{array}\right)(1-\epsilon)^{k} \epsilon^{T-k} \\ & \leqslant \exp \left(-\frac{1}{2} T(1-2 \epsilon)^{2}\right) \end{aligned}
$$

## 重采样

自适应权重重置和组合

1. 随机采样：bagging
2. 带权采样：boosting



## 串行式

特点：强依赖 

代表性：boosting---权值逐渐变大



### Boosting 

#### 算法步骤

1. 给所有训练样例赋予相同的权重
2. 训练第一个基本分类器
3. 对分类错误的**测试样例**提高其权重
4. 用调整过的带权**训练集**训练第二个基本分类器
5. 重复上述过程

6. 对所有的基分类器进行加权组合

$$
H_{M}(x)=\operatorname{sign}\left(\sum_{m=1}^{M} \alpha_{m} h_{m}(x)\right)
$$

$h_m$是基分类器，$w_n^m$表示样本权重，n为样本数量，m为个体分类器数

- 对于分类错误的样本--提高其权重



#### 🍑Ada boosting

考试会考

##### 模型

二分类问题：

N个训练样本：$x_{n}(n=1, \ldots, N)$

每个训练样本的标签为$y_{n} \in\{-1,+1\}, \quad h_{m}(x) \in\{-1,+1\}$



##### 算法步骤：

1. 初始化每个训练样本的权重$w_n$: $w_{n}^{(1)}=1 / N$，均匀分配
2. 第一个基分类器开始训练，通过最小化误差函数$min L_{m}=\sum_{n=1}^{N} w_{n}^{(m)} I\left(h_{m}\left(x_{n}\right) \neq y_{n}\right)$，$I$为指示函数，训练分类器$h_m$的参数
3. 计算加权的分类错误率$\epsilon_{m}=\frac{\sum_{n=1}^{N} w_{n}^{(m)} I\left(h_{m}\left(x_{n}\right) \neq y_{n}\right)}{\sum_{n=1}^{N} w_{n}^{(m)}}$，错误率逐渐降低
4. 计算分类器权重$\alpha_{m}=\ln \frac{1-\epsilon_{m}}{\epsilon_{m}}$，分类权重逐渐增大
5. 更新样本权重$w_{n}^{(m+1)}=w_{n}^{(m)} \exp \left(\alpha_{m} I\left(h_{m}\left(x_{n}\right) \neq y_{n}\right)\right)$
6. 使用$H_M(x) = sign(\sum_{m=1}^M\alpha_m h_m(x))$



#### Error model



## 并行式

不存在强依赖

### 决策树





# 贝叶斯分类器

x样本，c为标签

判别式模型：

生成式模型：

GAN：对抗生成网络

DCGAN：

## 模型

基于贝叶斯公式的后验概率：
$$
\begin{aligned} P\left(C=c_{i} \mid \mathbf{X} = \mathbf{x}\right) &=\frac{P\left(\mathbf{X}=\mathbf{x} \mid C=c_{i}\right) P\left(C=c_{i}\right)}{P(\mathbf{X}=\mathbf{x})} \\ & \propto P\left(\mathbf{X}=\mathbf{x} \mid C=c_{i}\right) P\left(C=c_{i}\right) \\ &  \text { for } i=1,2, \cdots, L \end{aligned}
$$
## 贝叶斯分类

$$
\underset{c_{i} \in C}{\operatorname{argmax}} {P\left(x_{1}, x_{2}, \ldots, x_{p} \mid c_{j}\right) P\left(c_{j}\right)}
$$
### 朴素贝叶斯分类

朴素条件：对已知类别，假设所有属性互相对立
$$
\begin{aligned} P\left(X_{1}, X_{2}, \cdots, X_{p} \mid C\right) &=P\left(X_{1} \mid X_{2}, \cdots, X_{p}, C\right) P\left(X_{2}, \cdots, X_{p} \mid C\right) \\ &=P\left(X_{1} \mid C\right) P\left(X_{2}, \cdots, X_{p} \mid C\right) \\ &=P\left(X_{1} \mid C\right) P\left(X_{2} \mid C\right) \cdots P\left(X_{p} \mid C\right) \end{aligned}
$$
朴素贝叶斯分类器模型（联合转化为连乘）：
$$
\arg \max _{c_{j} \in C} P\left(c_{j}\right) \prod_{i=1}^{P} P\left(x_{i} \mid c_{j}\right)
$$
需要估计：

- 先验$𝑃 (𝐶 = 𝑐_𝑗 )$
- 每个属性的条件概率$𝑃(𝑥_𝑖|𝑐_𝑗)$

#### 避免0概率问题

若某个属性值在训练集中没有与某个类同时出现过，则基于频率的概率估计将为零。

修正：在分母上+属性取值数目，分子加上类的个数
$$
\hat{P}\left(X_{\mathrm{i}}=x \mid C=c_{j}\right)=\frac{N\left(X_{\mathrm{i}}=x \mid C=c_{j}\right)+1}{N\left(C=c_{j}\right)+\left|X_{i}\right|}
$$
### 高斯朴素贝叶斯分类器

#### 高斯分布

$$
\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})=\frac{1}{(2 \pi) \mathbf{P} / 2} \frac{1}{|\boldsymbol{\Sigma}|^{1 / 2}} \exp \left\{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right\}
$$
一维Gaussian：

均值和方差的极大似然估计值分别是样本的均值及样本的方差
$$
\mu=\frac{1}{n} \sum_{i=1}^{n} x_{i}, \quad \sigma^{2}=\frac{1}{n} \sum_{i=1}^{n}\left(x_{i}-\mu\right)^{2}
$$
多维 Gaussian：
$$
\mu=\frac{1}{n} \sum_{i=1}^{n} x_{i}, \quad \Sigma=\frac{1}{n} \sum_{i=1}^{n}\left(x_{i}-\mu\right)\left(x_{i}-\mu\right)^{T}
$$
### 高斯朴素贝叶斯分类器

$$
\underset{C}{\operatorname{argmax}} P(C \mid X)=\underset{C}{\operatorname{argmax}} P(X, C)=\underset{C}{\operatorname{argmax}} P(X \mid C) P(C)
$$

$$
\hat{P}\left(x_{i} \mid C=c_{j}\right)=\frac{1}{\sqrt{2 \pi} \sigma_{i j}} \exp \left(-\frac{\left(x_{i}-\mu_{i j}\right)^{2}}{2 \sigma_{i j}^{2}}\right)
$$

x样本，c为标签

由X得到高斯分布的均值和方差，后代入高斯分布

$P(密度\mid好瓜) = P(密度\mid好瓜)*P(含糖率\mid好瓜)$



朴素：用所有样本

非朴素：联合等于两个乘积

$共有 L×(p+p× (p+1)/2)个参数$

$\sum:p×p$

$\mu:p$

朴素高斯必要性：估计的参数量减少



逻辑回归决策面：$\theta^TX = 0$

高斯贝叶斯决策面：

分到l类和k类的概率相等
$$
\log P\left(c_{k} \mid x\right)-\log P\left(c_{l} \mid x\right)=0
$$

用贝叶斯公式展开
$$
\log \frac{P\left(C_{k} \mid X\right)}{P\left(C_{1} \mid X\right)}=\log \frac{P\left(X \mid C_{k}\right)}{P\left(X \mid C_{l}\right)}+\log \frac{P\left(C_{k}\right)}{P\left(C_{l}\right)}
$$
其中
$$
\log P\left(x \mid c_{k}\right)=-\frac{1}{2}\left(\mathrm{x}-{\mu}_{k}\right)^{T} {\sum_{k}}^{-1}\left(x-\mu_{k}\right)-\log \left|\Sigma_{k}\right|^{\frac{1}{2}}
$$

$$
\begin{array}{l}\log \frac{P\left(c_{k} \mid \mathrm{x}\right)}{P\left(c_{l} \mid x\right)} \\ =\frac{1}{2}\left(\mathrm{x}-\mu_{l}\right)^{T} \Sigma_{l}{ }^{-1}\left(x-\mu_{l}\right)-\frac{1}{2}\left(\mathrm{x}-\mu_{k}\right)^{T} \Sigma_{k}{ }^{-1}\left(x-\mu_{k}\right)+\log \frac{\left|\Sigma_{l}\right|^{\frac{1}{2}}}{\left|\Sigma_{k}\right|^{\frac{1}{2}}}+\log \frac{\pi_{k}}{\pi_{l}}\end{array}
$$

假设每一类的协方差矩阵均相同，

$$
\sum_{\boldsymbol{k}}=\sum, \forall \boldsymbol{k}
$$

$$
\Sigma_{j}=\left[\begin{array}{ccc}\sigma_{1 j}^{2} & \cdots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \cdots & \sigma_{p j}^{2}\end{array}\right]
$$

决策函数可以从x的二次转为一次函数
$$
\begin{aligned}=& \log \frac{\pi_{k}}{\pi_{l}}+\frac{1}{2}\left(\mathrm{x}-\mu_{l}\right)^{T} \Sigma^{-1}\left(x-\mu_{l}\right)-\frac{1}{2}\left(\mathrm{x}-\mu_{k}\right)^{T} \Sigma^{-1}\left(x-\mu_{k}\right) \\=& \frac{\log \frac{\pi_{k}}{\pi_{l}}+\frac{1}{2} \mu_{l}^{T} \Sigma^{-1} \mu_{l}-\frac{1}{2} \mu_{k}^{T} \Sigma^{-1} \mu_{k}}{\mathrm{~b}}+x^{T} \frac{\Sigma^{-1}\left(\mu_{k}-\mu_{l}\right)}{\mathrm{a}} \end{aligned}
$$
决策边界：
$$
x^{T} \mathrm{a}+b=0
$$
要估计的参数个数：$L×(p+p× (p+1)/2)$

若 $a_{0}+\sum_{i=1}^{p} a_{j} x_{j}>0$，将𝑥的标签置为𝑐1，否则将其标签 置为𝑐2



## LDA决策面

通过假设每一类具有的相同协方差矩阵，得到一种经典 的线性学习方法：线性判别分析（Linear Discriminant Analysis, LDA）

线形决策面
$$
\begin{array}{l}=\log \frac{\pi_{k}}{\pi_{l}}+\frac{1}{2}\left(\mathrm{x}-\mu_{l}\right)^{T} \Sigma^{-1}\left(x-\mu_{l}\right)-\frac{1}{2}\left(\mathrm{x}-\mu_{k}\right)^{T} \Sigma^{-1}\left(x-\mu_{k}\right) \\ =\log \frac{\pi_{k}}{\pi_{l}}+\frac{1}{2} \mu_{l}^{T} \Sigma^{-1} \mu_{l}-\frac{1}{2} \mu_{k}^{T} \Sigma^{-1} \mu_{k}+x^{T} {\Sigma^{-1}\left(\mu_{k}-\mu_{l}\right)}\end{array}
$$

### 参数估计

先验：
$$
\widehat{P}\left(C=C_{j}\right)=\frac{N\left(C=c_{j}\right)}{N}
$$
均值：第j个高斯分布的均值
$$
\mu_{j}=\frac{1}{N\left(C=c_{j}\right)} \sum_{X \in c_{j}} X
$$
方差：
$$
\Sigma=\frac{1}{N} \sum_{c_{j} \in C} \sum_{X \in c_{j}}\left(X-\mu_{j}\right)\left(X-\mu_{j}\right)^{T}
$$
🍑求决策边界

已知相应的参数：
$$
\begin{array}{l}\begin{array}{l}* \pi_{1}=\pi_{2}=0.5 \\ * \mu_{1}=(0,0)^{T}, \mu_{2}=(2,-2)^{T}\end{array} \\ * \Sigma=\left(\begin{array}{cc}1.0 & 0.0 \\ 0.0 & 0.5625\end{array}\right)\end{array}
$$
则代入上述公式求得决策边界：
$$
\text { *Decision boundary: } 5.56-2.00 x_{1}+3.56 x_{2}=0.0
$$

Loss function

| 逻辑回归         | LDA               |
| ---------------- | ----------------- |
| 无𝑥 的分布假设： | 假设𝑥服从高斯分布 |

逻辑回归--判别式
$$
\begin{array}{l}{L = P(c \mid x ; \theta)}=\left(f_{\theta}(x)\right)^{c}\left(1-\frac{f_{\theta}}{\theta}(x)\right)^{1-c} \\ \theta^{(0)} \quad \theta^{(1)}:=\theta^{(0)}+\left.\alpha^{+\frac{e^{-\theta^{T}-x}}{\partial \theta}}\right|_{\theta^{(0)}}\end{array}
$$
LDA--生成式
$$
\begin{array}{l}P\left(x \mid c_{k}\right) \sim M\left(\mu_{k}, \Sigma_{h}\right) \\ =\frac{1}{(2 \pi)^{\frac{p}{2}} \mid \Sigma_{k} \cdot \frac{1}{2}} \exp -\frac{1}{2}\left(\mathrm{x}-\mu_{k}\right)^{T} \sum_{k}^{-1}\left(x-\mu_{k}\right) \\ \text { 其中 } \Sigma_{k}=\Sigma, \forall k . \quad \mu_{k}=\frac{1}{1} \sum_{\frac{2}{2}} x_{i} \\ \text { 决策边界 线性 } \sum_{k}=\frac{1}{h} \overline{2}\end{array}
$$

高斯朴素贝叶斯决策面




# K-NN分类算法

K-近邻 （K-nearest neighbors）

对一个未知样本进行分类：

1. 计算未知样本与标记样 本的距离

2. 确定 k 个近邻 

3. 使用近邻样本的标签确定目标的标签：

   例如， 将其划分到 k个样本中 出现最频繁的类

没有模型，没有error model

KNN回归



# 马尔可夫链

Markov模型（链）

## 贝叶斯网

有向无环图

特征变量之间的













![image-20211108090515589](/Users/wangjing/Library/Application Support/typora-user-images/image-20211108090515589.png)

1步转移
$$
\begin{array}{l}v_{t+1}(j)=P\left(X_{t+1}=j\right) \\ =\sum_{i=1}^{K} P\left(X_{t}=i\right) P\left(X_{t+1}=j \mid X_{t}=i\right)=\sum_{i=1}^{K} v_{t}(i) \mathrm{A}_{i j}=v_{t} \mathrm{~A}(:, \mathrm{j})\end{array}
$$
n步转移
$$
P({\mathrm{X}_{1}}=i_{1}, \ldots, \mathrm{X}_{T}=i_{T})=\pi_{i_{1}} \prod_{t=2}^{T} A_{i_{t-1} i_{t}}
$$

## 平稳分布

平稳分布：对于一个Markov链，给定初始状态分布 𝑣1 = 𝜋 = 𝑃 𝑋1 = 1 , ... , 𝑃 𝑋1 = 𝐾 ，利用状态转移 公式𝑣𝑡+1=𝑣𝑡A，经过一定次数的迭代之后，若能达到 ෤ 𝑣= ෤ 𝑣A 则称Markov链达到了平稳分布。 一旦进入平稳分布，在之后的任意时刻的概率分布 永远为 ෤ 𝑣，马尔可夫链处于稳定状态 稳定状态： ෤ 𝑣经过A转移后仍然是 ෤ 𝑣

应用

1. 句子补全
2. 网页排序：page-rank

damping look 解决断链问题

🍑平稳分布的计算

## page-rank

PageRank认为某个网页的重要性有两个因素决定：指 向网页的链接数量以及输出网页的链接数量。 

超链接的个数和质量

数量假设和质量假设

若网页𝑗到网页𝑖有边，则令 $𝐿_{𝑖𝑗} = 1$，否则 $𝐿_{𝑖𝑗} = 0$。因 此， $𝑗$的输出链接为，出链的个数（有向图出度）
$$
c_{j}=\sum_{i=1}^{N} L_{i j}
$$
指向网页的链接越多权重越大，而输出网页的链接越多权 重越小
$$
p_{i}=(1-d)+d \sum_{j=1}^{N}\left(\frac{L_{i j}}{c_{j}}\right) p_{j}
$$
$p_i$为为网页重要性，$c_j$表示网页j对网页i的重要性的程度



🍑参数估计计算

MLE最大似然估计for Markov chain



## HMM隐马尔可夫链

HMM三个问题

1. 评估问题：概率计算问题

   估计模型下观测序列出现的概率

2. 解码问题：状态预测问题

   给定模型参数和一个观测序列，推断隐状态 序列

3. 学习问题：参数估计问题

   给定多个观测数据Y，估计一组参数

![image-20211108110010355](/Users/wangjing/Library/Application Support/typora-user-images/image-20211108110010355.png)

常规方法：遍历--复杂度指数级



# 非监督学习
--压缩思想
1. 纵向结构--聚类
2. 横向结构--降维度


线形：
非线形：

## 聚类
clustering---簇内距小，簇间距大
簇的定义
数据表示：向量空间
相似性/距离：欧氏距离
簇的个数：数据驱动，自己识别出来
聚类算法：划分聚类算法，层次聚类算法
算法的收敛性：收敛速度

层次式聚类算法
1. 自顶向上：聚合
2. 自顶向下：分裂

⭐️划分式聚类算法
1. K-means
2. GMM（高斯混合模型）



## K-means

### 模型

算法步骤：

输入：数据N个样本，簇的个数指定为K

1. 初始化：随机选择K个数据点作为相应的簇中心{}
2. 迭代：
   1. 对每一个样本西交进行归簇，距离哪个聚类中心最近，则讲其归为哪一簇
   $x_{j} \in C_{i} \Leftrightarrow \min _{t=1, \ldots, K}\left\{\left\|x_{j}-\mu_{t}\right\|\right\}=\left\|x_{j}-\mu_{i}\right\|$
   2. 重新计算每个簇的均值（簇中心）$\mu_{i}=\frac{1}{\left|C_{i}\right|} \sum_{x_{j} \in C_{i}} x_{j}$
3. 终止迭代：簇中心不发生改变时

输出：簇中心

目标函数：簇内样本到簇中心的平方和距离最小
$$
\operatorname{argmin}_{C, \mu} \sum_{i=1}^{K} \sum_{x_{j} \in C_{i}}\left\|x_{j}-\mu_{i}\right\|_{2}^{2}
$$
非凸函数，NP-hard

解决之道：迭代优化（交替优化：固定一组变量值去优化另一组变量值）

• 初始化K个簇中心:𝜇 = {𝜇1, 𝜇2, ... , 𝜇𝐾} • 迭代进行以下优化

• 更新簇成员:固定𝜇，优化𝐶 • 更新簇中心:固定𝐶，优化𝜇



#### 算法复杂度：

迭代次数:假设迭代 𝑙 步算法收敛。因此总的计算复杂度

为 O(𝑙 𝐾np)
 由于𝐾和𝑙通常都远远小于n，可认为是关于n 的线性复杂度

#### 初值对算法的影响：

通过启发式方法选择好的初值:例如要求种子点之间有较大的距离 尝试多个初值，选择平方误差和最小的一组聚类结果

#### 聚类数目K的影响：

手肘法:目标函数的值和 k 的关系图是一个手肘的形状，而这个肘部 对应的k值就是数据的最佳聚类数。k=2时，对应肘部，故选择 k值为2

#### 局限性

不适合对形状不是超维椭圆体(或超维球体)的数据



## K-means延伸

层次-kmeans

乘积量化：就是划分数据集，然后分别划分出聚类中心，然后向量乘积

128*n

n个样本训练256个cluster：聚类中心

划分样本集做完笛卡尔乘积之后的维数和用原始样本求中心的维数一样吗？



# PCA

## 标准PCA

线形PCA

### 三种建模思想

PCA 求解角度

1. 最大投影方差
2. 最小投影距离
3. 奇异值分解(SVD)

#### 最大投影方差

信息（方差）能尽可能大的保持

#### 最小投影距离

投影数据与原数据的之间的最小平方距离尽可能小

目标函数：
$$
\mathbf{w}_{1}=\arg \max _{|\mathbf{w}|-1} \frac{1}{m} \sum_{i=1}^{m}\left\{\left(\mathbf{w}^{T} \mathbf{x}_{i}\right)^{2}\right\} \quad \operatorname{Var}(X)=E\left\{[X-E(X)]^{2}\right\}
$$

$$
L=-w^{\top} A w+\lambda\left(w^{\top} w-1\right)\\
\frac{\partial L}{\partial w}=-2 \cdot A w+2 \lambda w=0 \Rightarrow A w=\lambda w \Rightarrow w^{\top} A w=\lambda
$$

 w1协方差矩阵的最大特征值对应的特征向量
$$
\max _{W \in R^{p * k}} \operatorname{tr}\left(W^{T}\left(\frac{1}{m} X X^{T}\right) W\right), W^{T} W=I_{k}
$$
 PCA主方向 = 数据协方差矩阵的特征向量 • 更大特征值对应的特征向量更加重要
降维结果

$$
Z=W^{T} X
$$
重构结果
$$
\widehat{\mathrm{X}}=W Z=W W^{T} X
$$


目标:计算数据k个主方向

- 第一步:数据居中
- 第二步:计算居中数据的协方差矩阵
- 第三步:计算协方差矩阵最大k个特征值对应的特征 向量，组成矩阵

- 输出降维结果

- 问题:

  • 第k个主成分的方差是多少? 

  • k 选择多大
$$
  \begin{array}{l}w_{k}^{T}\left(\frac{1}{M} \sum_{i=1}^{M} x_{i} x_{i}^{T}\right) w_{k} \\ =w_{k}^{T} \lambda_{k} w_{k}=\lambda_{k} w_{k}^{T} w_{k}=\lambda_{k}\end{array}
$$
K的选择



### 奇异值分解SVD
$$
A=U \Sigma V^{T}
$$


#### PCA应用-数据预处理

数据白化(Whitening)操作

使用PCA，可以同时去除变量之间的线性关系以及对数据进行归一化:

- 假设数据的协方差矩阵为S
$$
  S=\frac{1}{m} \sum_{i=1}^m(x_{i}-\bar{x})(x_{i}-\bar{x})^{T}
$$

-  利用$W^{T} S W=\Lambda$定义一个变换
$$
y_{i}=\Lambda^{-\frac{1}{2}} W^{T}\left(x_{i}-\bar{x}\right)
$$

 则y的均值为0，协方差为单位矩阵。

## 概率PCA

## 核PCA

### 步骤

1. 输入
2. 构造Gram矩阵
3. 对高维数据去中心化
4. 对K进行特征分解
5. 计算x的低维表示

## LLE

Locally Linear embedding 局部线性嵌入

LLE关注于降维时保持样本局部的线性特征，由于LLE在降维时保持了样本的局部特征

1. 找最近邻：欧氏距离
2. 重构：重构系数之和=1

$$
\begin{aligned} \varepsilon(W) &=\sum_{i=1}^{N}\left\|x_{i}-\sum_{j=1}^{k} W_{i j} x_{i j}\right\|^{2} =\sum_{i=1}^{N}\left\|\sum_{j=1}^{k} W_{i j}\left(x_{i}-x_{i j}\right)\right\|^{2} \end{aligned}
$$
$W_{ij}$求解方法：拉格朗日乘数法
$$
L\left(W_{i}\right)=W_{i}^{T} Z_{i} W_{i}+\lambda\left(W_{i}^{T} 1_{k \times 1}-1\right)
$$

$$
2 Z_{i} W_{i}+\lambda 1_{k \times 1}=0, \text { 即 } W_{i}=-\frac{\lambda}{2} Z_{i}^{-1} 1_{k \times 1}
$$



3. 低维嵌入🍑



