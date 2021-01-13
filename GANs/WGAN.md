## Wasserstein GAN

### 1	引言

​		本文关注的问题是无监督学习。主要来说，学习概率分布是什么意思？这个问题的经典回答是概率密度。这通常是通过定义一个参数密度$(P_{\theta})_{\theta \in \mathbb{R}_d}$族，然后找到一个使我们的数据的似然性最大化的方法来完成的：如果我们有真实的数据示例$\{x^{(i)}\}_{i=1}^m$，我们要解决问题

$$\max_{\theta \in \mathbb{R}^d} \frac{1}{m}\sum_{i=1}^m \log P_{\theta}(x^{(i)})$$

如果真实数据分布$\mathbb{P}_r$允许一个密度，并且$\mathbb{P}_{\theta}$为参数化密度$P_{\theta}$的分布，那么，渐近地，者等于最小化Kullback-Leibler散度$KL(\mathbb{P}_r \| \mathbb{P}_{\theta})$。

​		为此，我们需要模型密度$P_{\theta}$存在。在处理由低维流形支持的分布的相当普遍的情况下，情况并非如此。这样，模型流形和真实分布的支撑不太可能具有不可忽略的交点（参见[1]），这意味着KL距离是不确定的（或仅是有限的）。

​		典型的补救措施是在模型分布中添加一个噪声项。这就是经典机器学习文献中描述的几乎所有生成模型都包含噪声分量的原因。在最简单的情况下，为了覆盖所有示例，我们假定带宽较高的高斯噪声。众所周知，例如，在图像生成模型的案例中，这些噪声衰减样本的质量，并使它们变模糊。例如，在最近的论文中，当像素已经被归一化到$[0,1]$范围内时，在对生成的图像中的每个像素最大化似然时，添加到模型的噪声的最佳标准偏差约为0.1。这是非常高的噪声，以至于当论文报告其模型样本时，他们没有添加噪声项来报告似然数。换句话说，添加的噪声项对于该问题显然是不正确的，但是使最大似然法起作用是必需的。

​		无需估计可能不存在的 $\mathbb{P}_r$ 的密度，我们可以定义具有固定分布$p(z)$的随机变量$Z$并将其传递给参数函数$g_{\theta}:\mathcal{Z} \rarr \mathcal{X}$（通常是某种神经网络），可以按照一定的分布$\mathbb{P}_{\theta}$直接生成样本。通过变化$\theta$，我们可以改变这种分布，并使它接近真实数据分布$\mathbb{P}_{r}$。这有两种用法。首先，与密度不同，这种方法可以表示限制在低维流形上的分布。其次，与学习密度的数值相比，轻松生成样本的能力通常更有用（例如，在考虑给定输入图像的输出图像的条件分布时，例如在图像超分辨率或语义分割中）。一般而言，在给定任意高维密度的情况下，很难通过计算生成样本[16]。

​		Variational Auto-Encoders（VAEs）和GANs是这种方法的著名示例。因为VAEs关注示例的近似似然，它们共享标准模型的局限性，因此需要增加噪声项。GANs在目标函数（包括Jensen-Shannon和$f$散度以及一些组合）的定义上更加灵活。另一方面，训练GAN众所周知，它精致且不稳定，[1]中给出理论上研究的原因。

​		本文中，我们将注意力集中在测量模型分布和实际分布有多接近的各种方法上，或者等效地，在定义距离或散度$\rho(\mathbb{P}_\theta, \mathbb{P}_r)$的各种方法上。**这种距离之间最根本的区别是它们对概率分布序列收敛的影响。**当且仅当存在一个分布$\mathbb{P}_{\infin}$使得$\rho(\mathbb{P}_r, \mathbb{P}_{\infin})$倾向零时，分布序列$(\mathbb{P}_t)_{t \in \mathbb{N}}$收敛。非正式地，**距离$\rho$在使分布序列更易于收敛时会导致拓扑结构变弱**。第2节阐述了流行的概率距离在这方面的差异。

​		为了优化参数$\theta$，当然希望以使映射$\theta \rarr \mathbb{P}_\theta$连续的方式定义模型分布$\mathbb{P}_{\theta}$。连续意味着，当参数序列$\theta_t$收敛到$\theta$时，分布$\mathbb{P}_{\theta_t}$也收敛到$\mathbb{P}_{\theta}$。但是，必须记住，分布$\mathbb{P}_t$收敛的概念取决于我们计算分布之间距离的方式。这种距离越弱，这种分布越容易定义$\theta$空间到$\mathbb{P}_{\theta}$空间连续映射，因为它更易于这种分布收敛。我们关心映射$\theta \rarr \mathbb{P}_{\theta}$连续的主要原因如下。如果$\rho$为两个分布之间距离的符号，那么我们希望有一个损失函数$\theta \rarr \rho(\mathbb{P}_\theta, \mathbb{P}_r)$是连续的，这等效于使用分布$\rho$之间的距离使映射$\theta \rarr \mathbb{P}_\theta$连续。

​		本文的贡献如下：

- 在第2节中，与学习分布情况下使用的流行概率距离和散度相比，我们提供了有关Earth Mover（EM）距离行为的综合理论分析。
- 第3节中，我们定义一种GAN形式，称为Wasserstein-GAN，其最小化EM距离的合理及有效近似，并且我们理论上证明对应的优化问题时合理的。
- 第4节中，我们从经验上证明WGAN解决了GAN的主要训练问题。特别地，WGAN不需要在鉴别器和生成器的训练上保持仔细的平衡，也不需要对网络架构进行精心设计。大大减少了GAN中典型的模式下降现象。WGAN的最引人注目的实用优势之一是能够通过将判别器训练到最佳状态来连续估算EM距离。绘制这些学习曲线不仅对调试和超参数搜索很有用，而且与观察到的样本质量也具有显着的相关性。

### 2    Different Distances

​		现在，我们介绍我们的符号。令$\mathcal{X}$为一个紧凑度量集（例如图像空间$[0,1]^d$），并令$\sum$为$\mathcal{X}$的所有Dorel子集的集合。令$\mathbf{Prob}(\mathcal{X})$表示定义在$\mathcal{X}$上的概率测量空间。现在，我们可以定义两个分布$\mathbb{P}_r,\mathbb{P}_g\in\mathbf{Prob}(\mathcal{X})$之间的初步距离和散度：

- _Total Variation (TV) 距离_

  $$\delta(\mathbb{P}_r, \mathbb{P}_g) = \sup_{A\in\sum}|\mathbb{P}_r(A), \mathbb{P}_g(A)|.$$

- _Kullback-Leibler (KL) 散度_

  $$KL(\mathbb{\mathbb{P}_r\|\mathbb{P}_g})=\int\log(\frac{P_r(x)}{P_g(x)})P_r(x)d\mu(x),$$

  其中$\mathbb{P}_r$和$\mathbb{P}_g$假设是绝对连续，因此相对于$\mathcal{X}$上定义的相同度量$\mu$允许密度。$KL$散度是著名的不对称，当存在$P_g(x)= 0$且$P_r(x)> 0$时可能会引起注意。

- _Jensen-Shannon_（JS）散度

  $$JS(\mathbb{P}_r, \mathbb{P}_g) = KL(\mathbb{P}_r \| \mathbb{P}_m) + KL(\mathbb{P}_g \| \mathbb{P}_m),$$

  其中$\mathbb{P}_m$为混合$(\mathbb{P}_r + \mathbb{P}_g) / 2$。由于我们可以选择$\mu = \mathbb{P}_m$，所以这种散度是对称的，并且始终是确定的。

- _Earth-Mover_（EM）距离或在Wasserstein-1

  $$W(\mathbb{P}_r, \mathbb{P}_g) = \inf_{\gamma \in \prod(\mathbb{P}_r,\mathbb{P}_g)}\mathbb{E}_{(x,y)\sim \gamma}[\|x-y\|],\tag{1}$$

  其中$\prod(\mathbb{P}_r,\mathbb{P}_g)$表示边界分别为$\mathbb{P}_r$和$\mathbb{P}_g$的所有联合分布$\gamma(x,u)$的集合。直觉上，$\gamma(x,y)$表示必须将多少“质量（mass）”从$x$传输到$y$，才能将分布$\mathbb{P}_r$转换为分布$\mathbb{P}_g$。 

​        以下示例说明了概率分布的简单序列如何在EM距离下收敛，而在上面定义的其他距离和散度下不收敛。

**Example 1**	（Learning parallel lines）。令$Z \sim U[0,1]$为单位间隔上的均匀分布。令$\mathbb{P}_0$为$(0, Z) \in \mathbb{R}^2$（0在x轴上，随机变量$Z$在y轴上）的分布，该分布均匀分布在穿过原点的垂直直线。现在，令$g_{\theta}(z) = (\theta, z)$，$\theta$为单个实参。在这种情况下，容易得到

- $$W(\mathbb{P}_0, \mathbb{P}_\theta) = |\theta|,$$
- $$JS(\mathbb{P}_0, \mathbb{P}_{\theta}) = \begin{cases}\log 2 &\mbox{ if } \theta \ne 0,\\0 &\mbox{ if } \theta = 0,\end{cases}$$
- $$KL(\mathbb{P}_\theta\|\mathbb{P}_0) = KL(\mathbb{P}_0\|\mathbb{P}_\theta) = \begin{cases}+\infin &\mbox{ if } \theta \ne 0,\\ 0 &\mbox{ if } \theta = 0,\end{cases}$$
- $\delta(\mathbb{P}_0, \mathbb{P}_\theta) = \begin{cases}1 &\mbox{ if } \theta \ne 0, \\ 0 &\mbox{ if } \theta = 0.\end{cases}$

当$\theta_t \rarr 0$，在EM距离下，序列$(\mathbb{P}_{\theta_t})_{t \in \mathbb{N}}$收敛到$\mathbb{P}_0$，但是在JS、KL、逆KL或TV距离下，不会收敛。图1说明了EM和JS距离的情况。 

![fig1](images/WGAN/fig1.png)

​		示例1给出了一个案例，其中我们可以通过对EM距离进行梯度下降来学习低维流形上的概率分布。这不能用其他距离和散度来完成，因为所得的损失函数甚至不是连续的。 尽管此简单示例的特征是具有不连续支撑的分布，但是当支撑在一组零度量中包含非空交集时，得出相同的结论。当两个低维流形在一般位置相交时，情况就是这样[1]。

**假设1**	令$g: \mathcal{Z} \times \mathbb{R}^d \rarr \mathcal{X}$为有限维度向量空间之间的局部Lipschitz。我们将$g_\theta(z)$表示为它在坐标$(z, \theta)$上的评估。对于$\mathcal{Z}$上的某个概率分布$p$，如果存在局部Lipschitz常量$L(\theta,z)$，使得

$$\mathbb{E}_{z \sim p}[L(\theta, z)] < + \infin$$

那么我们称$g$满足假设1。

​		因为Wasserstein距离远弱于JS距离，所以我们现在可以问$W(\mathbb{P}_r,\mathbb{P}_\theta）$是否是在温和假设下的连续损失函数。正如我们现在陈述和证明的那样，这是成立的。

**Theorem 1**	令$\mathbb{P}_r$为$\mathcal{X}$上的固定分布。令$Z$为另一个空间$\mathcal{Z}$上的随机变量（例如Gaussian）。令$g: \mathcal{Z} \times \mathbb{R}^d \rarr \mathcal{X}$为一个函数，其表示为$g_\theta(z)$，其中$z$为第一个坐标，$\theta$为第二个。令$\mathbb{P}_\theta$表示$g_\theta(Z)$的分布。那么，

1. 如果$g$在$\theta$上连续，那么$W(\mathbb{P}_r, \mathbb{P}_\theta)$也是如此。
2. 如果$g$是局部Lipschitz并且满足正则性假设1，则$W(\mathbb{P}_r, \mathbb{P}_\theta)$在任何地方都是连续的，并且几乎在所有地方都是可微的。
3. 对于Jensen-Shannon散度$JS(\mathbb{P}_r,\mathbb{P}_\theta)$和所有KL，语句1-2为假。

_证明：_见附录C。

**Theorem 2**	令$\mathbb{P}$为紧凑空间$\mathcal{X}$上的分布，$(\mathbb{P}_n)_{n \in \mathbb{N}}$为$\mathcal{X}$上的序列。然后，随着$n \rarr \infin$，有

1. 如下表达式等价
   - $\delta(\mathbb{P}_n, \mathbb{P}) \rarr 0$，其中$\delta$为总变化距离。
   - $JS(\mathbb{P}_n,\mathbb{P}) \rarr 0$，其中$JS$为Jensen-Shannon散度。
2. 如下表达式等价
   - $W(\mathbb{P}_n,\mathbb{P}) \rarr 0$。
   - $\mathbb{P}_n \xrightarrow{\mathcal{D}} \mathbb{P}$，其中$\xrightarrow{\mathcal{D}}$表示随机变量的分布的收敛性。
3. $KL(\mathbb{P}_n \| \mathbb{P}) \rarr 0$或$KL(\mathbb{P}\|\mathbb{P}_n) \rarr 0$隐含在（1）的表达式中。
4. （1）中的表达式隐含（2）中表达式。

_证明_ 见附录C。

​		这突显了以下事实：在学习由低维流形支持的分布时，KL、JS和TV距离不是明智的成本函数。但是，在该设置中，EM距离是明智的。 显然，这将使我们进入下一部分，在该部分我们将介绍优化EM距离的实用近似方法。

### 3	Wasserstein GAN

​		再次，定理2指出了一个事实，即优化后的$W(\mathbb{P}_r, \mathbb{P})$可能比$JS(\mathbb{P}_r, \mathbb{P})$具有更好的属性。但是，（1）中的无限是最难处理。另一方面，Kantorovich-Rubinstein 二重性（duality）告诉我们

$$W(\mathbb{P}_r, \mathbb{P}_\theta) = \sup_{\|f\|_{L}\le1} \mathbb{E}_{x \sim \mathbb{P}_r}[f(x)] - \mathbb{E}_{x\sim\mathbb{P}_\theta}[f(x)], \tag{2}$$

其中上界（supremum）为所有1-Lipschitz函数$f: \mathcal{X} \rarr \mathbb{R}$。注意，如果我们将$\|f\|_L\le1$替换为$\|f\|_L \le K$（对于一些常量$K$，考虑K-Lipschitz），那么我们以$K\cdot W(\mathbb{P}_r,\mathbb{P}_g)$结束。因此，如果我们有一个参数化的函数$\{f_w\}_{w\in\mathcal{W}}$系列，它们对于某些$K$都是K-Lipschitz，则可以考虑解决该问题

$$\max_{w \in \mathcal{W}} \mathbb{E}_{x \sim \mathbb{P}_r}[f_w(x)] - \mathbb{E}_{z \sim p(z)}[f_w(g_\theta(z))], \tag{3}$$

并且如果（2）中的上界达到某个$w \in \mathcal{W}$（一个很强的假设，类似于证明估计量的一致性时所假定的假设），则此过程将产生$W(\mathbb{P}_r, \mathbb{P}_\theta)$的计算，直到乘法常数。此外，我们可以考虑通过估算$\mathbb{E}_{z \in p(z)}[\bigtriangledown_\theta f_w(g_\theta(z))]$通过方程式（2）反向传播来对$W(\mathbb{P}_r,\mathbb{P}_\theta)$进行微分（再次，直至一个常数）。虽然这都是直觉，但我们现在证明此过程是在最佳假设下进行的。

**Theorem 3**	令$\mathbb{P}_r$为任意分布。令$\mathbb{P}_\theta$为$g_\theta(Z)$的分布，其中$Z$为密度为$p$的随机变量，而$g_\theta$为满足假设1的函数。然后，当两项都明确定义时，有这个问题的解$f: \mathcal{X} \rarr \mathbb{R}$

$$\max_{\|f\|_{L}\le1}\mathbb{E}_{x \sim \mathbb{P}_r}[f(x)] - \mathbb{E}_{x \sim \mathbb{P}_\theta}[f(x)]$$

并且，我们有

$$\bigtriangledown_\theta W(\mathbb{P}_r,\mathbb{P}_\theta) - \mathbb{E}_{z \sim p(z)}[\bigtriangledown_\theta f(g_\theta(z))]$$

​		现在出现的问题是找到解决方程（2）中的最大化问题的函数$f$。