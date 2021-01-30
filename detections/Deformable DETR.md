## Deformable DETR: Deformable Transformers for End-to-End Object Detection

### 摘要

​		最近提出DETR来消除目标检测中许多手工设计的组件，同时表现出良好的性能。但是，由于Transformer注意力模块在处理图像特征时的局限，它遭遇收敛速度慢以及有限的特征空间分辨率。Deformable DETR可以获得比DETR（尤其在小目标上）更好的性能，而且仅需DETR十分之一的训练epoch。在COCO基准上进行的大量实验证明了我们方法的有效性。

### 1	引言

​		现代目标检测器采用许多手工设计的组件，例如锚生成、基于规则的训练目标分配、NMS后处理。它们不完全是端到端的。最近，Carion等提出DETR来消除如此手工组件的需要，并构建了第一个完全端到端的目标检测器，获得非常具竞争性的性能。DETR利用简单的架构，通过结合CNN和Transformer encode-decoder。他们利用Transformer的强大的关系建模能力，在适当设计的训练信号下替换了手工制作的规则。

​		尽管有趣的设计和良好的性能，DETR有其自身的问题：（1）它需要的训练epoch远比已有目标检测器更长。例如，在COCO基准测试上，DETR需要500个epoch才能收敛，其比Faster R-CNN慢约10到20倍。（2）DETR在检测小型目标上，表现出相对较低的性能。现代目标检测器通常探索多尺度特征，其中在高分辨率特征图上检测小型目标。同时，高分辨率对于DETR来说是不可接受的复杂度。上述问题主要可归因于在处理图像特征图时Transformer组件的赤字。在初始时，注意力模块对特征图中的所有像素几乎都施加了相同的注意权重。要使注意力权重集中在稀疏的有意义位置上，就必须进行长时间的训练。另一方面，Transformer编码器中的注意力权重的计算为像素数的二次方。因此，处理高分辨率特征图具有很高的计算和存储复杂性。

​		在图像领域，变形卷积（Dai等2017）是一种强大而有效的机制，可以处理稀疏的空间位置。它自然而言地避免上述问题。而它缺乏元素关系建模机制，其是DETR成功的关键。

​		本文中，我们提出Deformable DETR，其缓解DETR较慢收敛和高复杂度问题。它结合了可变形卷积稀疏空间采样的最佳功能，以及Transformer的关系建模能力。我们提出_deformable attention module_，其关注一小组采样位置，作为所有特征图像素中主要关键元素的预滤波器。该模块可以自然地扩展以聚合多尺度特征，而没有FPN（Lin等2017a）的帮助。在Deformable DETR中，我们利用（多尺度）可变形注意力模块来替换处理特征图的Transformer注意力模块，如图1所示。

​		得益于其快速收敛，计算和存储效率，Deformable DETR为我们提供了利用端对端目标检测器变体的可能性。我们探索简单而有效的_iterative bounding box refinement_机制来提高检测性能。我们还尝试两阶段的Deformable DETR，其中也由Deformable DETR的变体生成区域提议，其进一步馈入解码器进行迭代边界框精炼。

​		在COCO基准上的大量实验证明我们的方法的有效性。与DETR相比，Deformable DETR可以以少10倍的训练epoch获得更好的性能（尤其对于小目标）。所提出的两阶段Deformable DETR变体可以进一步提高性能。

### 2	相关工作

**Efficient Attention Mechanism**	Transformers包含自注意力和cross-attention机制。Transformers最著名的担忧是大量关键元素上的时间和内存复杂性，其妨碍许多情况下的建模能力。最近，已许作出许多努力来处理这个问题（Tay等2020b），其可以大致分为三类。

​		第一类是在键上使用预定义的稀疏注意力模式。最直接的范式是将注意力模式限制到固定的局部窗口。大多数工作遵循这种范式。尽管讲注意力模式限制到局部近邻可以减少复杂度，但是丢失了全局信息。作为补偿Child等（2019）、Huang等（2019）、Ho等（2019）关注固定区间上的键元素以显著提高键上的感受野。Deltagy等（2020）、Ainslie等（2020）、Zaheer等（2020）允许少数特殊令牌访问所有关键元素。Zaheer等（2020）、Qiu等（2019）也添加一些预固定的注意力模式来直接关注键元素。

​		第二类是学习数据依赖的稀疏注意力。Kitaev等（2020）提出了一种基于位置敏感的哈希（LSH）的注意力，它把查询和关键元素都散列到不同的bin中。Roy等（2020）提出一种相似的思想，其中利用k-means找出最相关的键。Tay等人（2020a）学习了块置换以实现块状稀疏注意力。

​		第三类是探索自注意力中低秩特性。Wang等（2020）通过在尺寸维度而非通道维度上进行线性投影来减少键元素的数量。Katharopoulos等（2020）、Choromanski等（2020）通过核化近似重写了自注意力的计算。

​		在图像域中。高效的注意力机制仍限于第一类。尽管理论上减小了复杂度，但由于内存访问模式上的固有限制，Parmar等（2019）、Hu等（2019）承认如此方法在实现上远慢于具有相同FLOPs的传统卷积（至少慢3倍）。

​		另一方面，如Zhu等（2019a）讨论的，存在卷积变体（例如可变形卷积和动态卷积）也可以视为自注意力机制。尤其是，可变形卷积运算在图像识别上远比Transformer自注意力有效。同时，它却缺乏元素关系建模机制。

​		我们提出的可变形注意力模块受可变形卷积的启发，并属于第二类。它仅关注根据查询元素的特征预测的一小组固定的采样点。与Parmar等（2019）、Hu等（2019）不同，在相同的FLOPs下，可变形注意力仅略慢于传统卷积。

**Multi-scale Feature Representation for Object Detection**	目标检测仪的主要难题是在不同尺度高效地表示目标。现代目标检测器通常利用多尺度特征迁就这个问题。作为先驱工作之一，FPN提出自上而下的路径来组合多尺度特征。PANet在FPN之上进一步添加自下而上的路径。Kong等（2018）通过全局注意力操作来组合来自所有尺度的特征。Zhao等（2019）提出U形模块来融合多尺度特征。最近，提出NAS-FPN和Auto-FPN通过神经架构搜索来自动设计跨尺度连接。Tan等（2020）提出BiFPN，其为PANet的重复简化版本。我们提出的多尺度可变形注意力模块可以通过注意力机制自然而言地聚合多尺度特征，而无需特征金字塔网络的帮助。

### 3	回顾Transformers和DETR

**Multi-Head Attention in Transformers**	Transformers具有基于注意力机制的机器翻译网络架构。给定一个查询元素（例如输出语句中的目标单词）和一组键元素（例如输入语句中源单词），多头自注意力模块根据衡量查询-键对的兼容性的注意力权重自适应地聚合键内容。为了使该模型专注于来自不同表示子空间和不同位置的内容，将不同注意力头的输出与可学习的权重进行线性聚合。令$q \in \Omega_q$表示具有表示特征$\mathbf{z}_q \in \mathbb{R}^C$的查询元素，$k \in \Omega_k$表示具有表示特征$\mathbf{x}_k \in \mathbb{R}^C$的键元素，其中$C$为特征维度，$\Omega_q$和$\Omega_k$分别为查询和键元素集。然后，多头注意力计算为：

$$\mbox{MultiHeadAttn}(\mathbf{z}_q,\mathbf{x}) = \sum_{m=1}^M \mathbf{W}_m\big[\sum_{k \in \Omega_k} A_{mqk} \cdot \mathbf{W}_m'\mathbf{x}_k\big],\tag{1}$$

其中$m$表示注意力头部，$\mathbf{W}_m' \in \mathbb{R}^{C_v \times C}$和$\mathbf{W}_m \in \mathbb{R}^{C \times C_v}$为可学习的权重（默认情况下$C_v = C/M$）。注意力权重$A_{mqk} \propto \exp\big\{\frac{\mathbf{z}_q^T \mathbf{U}_m^T\mathbf{V}_m\mathbf{x}_l}{\sqrt{C_v}}\big\}$被规范化为$\sum_{k\in\Omega} A_{mqk} = 1$，其中$\mathbf{U}_m,\mathbf{V}_m \in \mathbb{R}^{{C_v}\times C}$也为可学习的权重。为了消除不同空间位置，表示特征$\mathbf{z}_q$和$\mathbf{x}_k$通常是元素内容和位置嵌入的串联/求和。

​		Transformers有两个已知问题。一是，Transformers在收敛之前需要长时间训练。假设查询和键元素的数量分别为$N_q$和$N_k$。通常，利用适当的参数初始化，$\mathbf{U}_m\mathbf{z}_q$和$\mathbf{V}_m\mathbf{x}_k$服从均值为0、方差为1的分布，当$N_k$很大时，其使注意力权重$A_{mqk}\approx \frac{1}{N_k}$。这将导致输入特征的模糊特征。因此，需要长时间训练调度使得注意力权重可以关注特定的键。在图像域，其中键元素通常是图像像素，$N_k$可以非常大，并且收敛很慢。

​		另一方面，使用众多查询和关键元素，多头注意力的计算和内存复杂度可能会非常高。式（1）的计算复杂度为$O(N_qC^2 + N_kC^2 + N_qN_kC)$。在图像域，其中查询和键元素都是像素。$N_q = N_k \gg C$，第三项主导复杂度，为$O(N_qN_kC)$。因此，多头注意力模块随着特征图的大小呈平方增长。

**DETR**	DETR构建在Transformer encoder-decoder架构之上，结合基于集合的匈牙利损失，它通过二分匹配对每个ground-truth边界框强制进行唯一的预测。 我们简要回顾一下网络架构，如下所示。

​		给定由CNN骨干提取的输入特征图$\mathbf{x} \in \mathbb{R}^{C \times H \times W}$，DETR利用标准的Transformer encoder-decoder架构讲输入特征图转换为目标查询集合的特征。然后在目标查询特征（由解码器生成）上添加3层前馈神经网（Feed-Forward Neural Network：FFN）和一层线性映射作为检测头部。FFN作为回归分支来预测边界框坐标$\mathbf{b} \in [0, 1]^4$，其中$\mathbf{b} = \{b_x, b_y, b_w, b_h\}$编码（相对于图像大小的）归一化的边界框中心坐标、边界框高和宽。线性映射作为分类分支来生成分类结果。

​		对于DETR中的Transformer编码器，查询元素和键元素都是特征图中的像素。输入为（具有编码位置嵌入的）ResNet特征图。令$H$和$W$分别表示特征图高和宽。自注意力的计算复杂度为$O(H^2W^2C)$，其随空间大小呈二次方增长。

​		对于DETR中的Transformer解码器，输入包含来自编码器的特征图和由可学习的位置嵌入表示的$N$个目标查询（例如$N=100$）。解码器中有两种类型的注意力模块，即cross-attention和self-attention模块。在cross-attention模块中，目标查询从特征图提取特征。查询元素为目标查询，键元素为编码器的输出特征图。其中$N_q = N$、$N_k = H \times W$，cross-attention的复杂度为$O(HWC^2 + NHWC)$。复杂度随特征图大小呈线性增长。在自注意力模块中，目标查询相互交互，同时捕获它们关系。其中，$N_q = N_k = N$，self-attention模块的计算复杂度为$O(2NC^2 + N^2C)$。中等数量的目标查询可以接受这种复杂性。

​		DETR是用于目标检测的极具吸引力的设计，从而消除了对许多手工设计组件的需求。但是，它也其自身的问题。这些问题主要归因于Transformer注意力在将图像特征图处理为键元素中的赤字：（1）DETR在检测小型目标时，性能相对较低。但是，高分辨率特征图将导致DETR的Transformer编码器中的自注意力模块的复杂性令人无法接受，该模块的复杂度随输入特征图的空间大小呈二次方增长。（2）与现代目标检测器相比，DETR需要更多的训练epoch才能收敛。这主要是因为注意力模块处理图像特征难以训练。例如，在初始时，cross-attention模块几乎是整个特征图的平均注意力。同时，在训练结束时，学习到的注意力图非常稀疏，其仅关注目标极点。似乎DETR需要很长的训练时间来学习注意力图中的重大变化。

### 4	Method

#### 4.1	Deformable Transformers for End-to-End Object Detection

**Deformable Attention Module**	在图像特征图上使用Transformer注意力的核心问题是，它会查看所有可能的空间位置。为了处理这个问题，我们提出_deformable attention module_。受可变形卷积的启发，可变形注意力模块仅关注参考点周围的键采样点的小型固定集合，而不论特征图的空间大小。通过为每个查询仅分配少量固定数量的键，收敛和特征空间分辨率问题可以得到缓解。

​		给定输入特征图$\mathbf{x} \in \mathbb{R}^{C \times H \times W}$，令$q$为具有内容特征$\mathbf{z}_q$和2维参考点$\mathbf{p}_q$的的查询元素的索引，可变形注意力特征计算为：

$$\mbox{DeformAttn}(\mathbf{z}_q,\mathbf{p}_q, \mathbf{x}) = \sum_{m=1}^M \mathbf{W}_m\big[\sum_{k=1}^K A_{mqk} \cdot \mathbf{W}_m'\mathbf{x}(\mathbf{p}_q + \Delta\mathbf{p}_{mqk})\big],\tag{2}$$

其中$m$索引注意力头，$k$索引采样键，$K$为全部的采样键的数量（$K \ll HW$）。$\Delta \mathbf{p}_{mqk}$和$A_{mqk}$表示分别表示第$m$个注意力头部第$k$个采样点的采样偏移和注意力权重。标量注意力权重$A_{mqk}$位于$[0,1]$，其通过$\sum_{k=1}^K A_{mqk}=1$归一化。$\Delta \mathcal{p}_{mqk}\in\mathbb{R}^2$为无约束范围的2维实数。因为$\mathbf{p}_q + \Delta\mathbf{p}_{mqk}$为小数，计算$\mathcal{x}(\mathbf{p}_q + \Delta \mathbf{p}_{mqk})$时，如Dai等（2017）使用双线性插值。在实现中，将查询特征$\mathbf{z}_q$馈入$3MK$个通道的线性映射算子，其中前$2MK$个通道编码采样偏移$\Delta \mathbf{p}_{mqk}$，余下的$MK$个通道馈入$\mbox{softmax}$算子以获得注意力权重$A_{mqk}$。

​	可变形注意力模块设计用于处理卷积特征图作为键元素。令$N_q$为查询元素的数量，当$MK$相对较小时，可变形卷积的复杂度为$O(2N_qC^2 + \min(HWC^2, N_qKC^2))$（细节见附录A.1）。当将其用于DETR编码器上是，其中$N_q = HW$，复杂度变为$O(HWC^2)$，其复杂度随空间大小呈线性变化。当将其用作DETR解码器的cross-attention模块时，其中$N_q = N$（$N$为目标查询数量），复杂度变为$O(NKC^2)$，其与空间大小$HW$无关。

**Multi-scale Deformable Attention Module**	大多数现代目标检测框架得益于多尺度特征。我们提出的可变形注意力模块可以自然而言地扩展到多尺度特征图。

​		令$\{\mathbf{x}^l\}_{l=1}^L$为输入的多尺度特征图，其中$\mathbf{x}^l \in \mathbb{R}^{C \times H_l \times W_l}$。令$\hat{\mathcal{p}}_q \in [0,1]^2$为每个查询元素$q$的参考点的归一化坐标，那么将多尺度可变形注意力为：

$$\mbox{MSDeformAttn}(\mathbf{z}_q, \hat{\mathbf{p}}_q,\{\mathbf{x}^l\}_{l=1}^L) = \sum_{m=1}^M \mathbf{W}_m\big[\sum_{l=1}^L\sum_{k=1}^K A_{mqk} \cdot \mathbf{W}_m'\mathbf{x}^l(\phi_l(\hat{\mathbf{p}}_q) + \Delta \mathbf{p}_{mlqk})\big],\tag{3}$$

其中$m$索引注意力头部，$l$索引输入图特征层，$k$索引采样点。$\Delta\mathbf{p}_{mlqk}$和$A_{mlqk}$分别表示第$l$特征层和第$m$个注意力头部中的第$k$个采样点的采样偏移和注意力权重。标量注意力权重$A_{mlqk}$由$\sum_{l=1}^L\sum_{k=1}^KA_{mlqk}=1$归一化。这里，为了尺度共识的简明，我们使用归一化坐标$\hat{\mathbf{p}}_q \in [0, 1]^2$，其中归一化坐标$(0,0)$和$(1,1)$分别表示图像的左上角和右下角。式（3）中的函数$\phi_l(\hat{\mathbf{p}}_q)$将归一化坐标$\hat{\mathbf{p}}_q$重新调到第$l$层输入特征图。多尺度可变形注意力与前面的单尺度版非常相似，除了从多尺度特征图上采样$KL$个点，而不是从单尺度特征图上采样$K$个点。

​		当$L=1$、$K=1$以及$\mathbf{W}_m' \in \mathbb{R}^{C_v \times C}$固定为恒等矩阵时，所提出的注意力模块将衰退为可变形卷积。可变形卷积设计用于单尺度输入，每个注意力头部，其仅关注一个采样点。然而，我们我们多尺度可变形卷积查看多尺度输入的多个采样点。所提出（多尺度）可变形注意力模块也可以被感知为Transformer注意力的有效变体，其中通过可变形采样位置引入预滤波机制。当采样点遍历所有可能位置时，所提出的注意力模块等价于Transformer注意力。

**Deformable Transformer Encoder**	我们将DETR中处理特征图的Transformer注意力模块替换为所提出的多尺度可变形注意力模块。编码器的输入和输出都为具有相同分辨率的多尺度特征图。在编码器中，我们从ResNet的阶段$C_3$到阶段$C_5$的输出特征图上提取多尺度特征图$\{\mathbf{x}\}_{l=1}^{L-1}$（$L=4$），其中$C_l$的分辨率为输入图像的$1/2^l$。最低分辨率的特征图$\mathbf{x}^L$通过在最终的$C_5$阶段上使用步长为2的$3 \times 3$卷积获得，表示为$C_6$。所有多尺度特征的通道都为$C=256$。值得注意的是没有使用FPN中top-down结构，因为我们所提出的多尺度可变形注意力自身可以交换多尺度特征图之间的信息。多尺度特征图的构建见附录A.2。5.2节中的实验表明添加FPN不会提高性能。

​		在编码器的多尺度可变形注意力的应用中，输出是具有与输入相同分辨率的多尺度特征图。键和查询元素都来自多尺度特征图。对于每个查询像素，参考点是其自身。为了确定每个查询像素位于哪个特征层，除了位置嵌入外，我们将尺度级嵌入（表示为$\mathbf{e}_l$）添加到特征表示。与固定编码的位置嵌入不同，尺度级嵌入$\{\mathbf{e}_l\}_{l=1}^L$被随机初始化，并与网络联合训练。

**Deformable Transformer Decoder**	在解码器中有cross-attention和self-attention模块。两种类型的注意力模块的查询元素都是目标查询。在cross-attention模块中，从特征图上提取目标查询，其中键元素为目标查询。在self-attention模块中，目标查询彼此交互，其中键元素是对象查询。由于我们提出的可变形注意力模块设计用于处理卷积特征图作为键元素，我们仅将每个cross-attention模块替换为多尺度可变形注意力模块，而余下的自注意力模块不会改变。对于每个目标查询，参考点$\hat{\mathbf{p}}_q$的二维归一化坐标是通过可学习的线性投影以及随后的S型函数从其mub 查询嵌入中预测的。

​		因为从参考点周围的图像特征上提取多尺度可变形注意力模块，我们让检测头将边界框预测为与参考点对应的相对偏移进一步降低了优化难度。参考点被用作边界框中心的初始猜测。检测头预测与参考点对应的相对偏移。详细细节见附录A.3。以这种方式，学习到的解码器注意力将与预测的边界框有强力的相关性，其也加速收敛。

​		通过利用可变形注意力模块替换DETR中Transformer注意力模块，我们建立有效而快速收敛的检测系统，称为Deformable DETR（见图1）。

#### 4.2 Additional Improvements and Variants for Deformable DETR

​		由于Deformable DETR的快速收敛、计算和内存效率，它开启我们利用端到端目标检测器不同变体的可能。由于有限的空间，这里，我们仅介绍这些改进和变体的核心思想。实现细节见附录A.4。

**Iterative Bounding Box Refinement**	受光流估计（Teed & Deng，2020）中开发的迭代精炼的启发，我们建立简单而有效的迭代边界框精炼机制来提高检测性能。这里，每个解码器基于前一层的预测精炼边界框。

**Two-Stage Deformable DETR**	在原始的DETR中，解码器中的目标查询与当前图像无关。受两阶段目标检测器的启发，我们探索Deformable DETR变体生成区域提议，以作为第一个阶段内容。生成的提议将馈入解码器作为目标查询进行进一步精炼，从而构成两阶段Deformable DETR。

​		在第一个阶段中，为了获得高召回提议，多尺度特征图中的每个像素作为一个目标查询。但是，将目标查询直接设置为像素会给解码器中的自注意力模块带来不可接受的计算和内存开销，其复杂度会随着查询数量的增加而平方增长。为了避免这个问题，我们移除解码器，并形成一个仅有编码器的Deformable DETR用于区域提议生成。其中，每个像素被分配为一个目标查询，其直接预测一个边界框。选择top scoring边界框作为区域提议。在将区域提议馈入第二阶段前，不使用NMS。

### 5	实验

**Dataset**	我们在COCO2017数据集上进行实验。在训练集上训练模型，在验证集和test-dev集上评估模型。

**实现细节**	ImageNet预训练的ResNet-50用作骨干进行消融。提取多尺度特征图，而没有FPN。默认情况下，可变形注意力中，$M=8$和$K=4$。可变形Transformer编码器的参数在不同特征层中共享。其他超参数设置和训练策略主要遵循DETR，除了损失权重为2的Focal Loss（Lin et al，2017b）用于边界框分类，并且增加了将目标查询的数量从100增加到300。我们还具有这些修改的DETR-DC5的性能以进行公平比较，表示为$\mbox{DETR-DC5}^+$。默认情况下，模型训练50个epoch，学习率在第40个epoch乘以0.1。遵循DETR，我们使用基学习为$2 \times 10^{-4}$、$\beta_1 = 0.9$、$\beta_2=0.999$、权重衰减为$10^{-4}$的Adam优化器训练模型。线性映射（用于预测目标查询参考点和采样偏移）的学习率乘以0.1。NVIDIA Tesla V100 GPU上评估运行时。

#### 5.1	与DETR的比较

![table1](images/Deformable%20DETR/table1.png)

​		如图1所示，与Faster R-CNN + FPN相比，DETR需要更多的训练epoch才能收敛，并且小目标上的检测性能更低。与DETR相比，Deformable DETR以少10倍的训练epoch获得更好的性能（尤其是小目标）。详细的收敛曲线见图2。借助于迭代边界框的精炼和两阶段范式，我们的方法可以进一步提高检测精度。

![fig2](images/Deformable%20DETR/fig2.png)

​		我们所提出的DETR与Faster R-CNN + FPN和DETR-DC5相当的FLOPs。但是运行时间远快（1.6倍）于DETR-DC5，仅比Faster R-CNN+FPN慢25%。DETR-DC5的速度问题主要源自Transformer attention的大量内存访问。我们所提出的可变形注意力可以缓解这个问题，但以无序的内存访问为代价。 因此，它仍然比传统卷积稍慢。

#### 5.2	Deformable Attention的消融研究

​		![table2](images/Deformable%20DETR/table2.png)

#### 5.3 与最佳方法的比较

![table3](images/Deformable%20DETR/table3.png)

### A	附录

#### A.1	可变形注意力的复杂度

​		假设查询元素的数量为$N_q$，在可变形注意力模块中（见式2），计算采样坐标偏移$\Delta \mathbf{p}_{mqk}$和注意力权重$A_{mqk}$的复杂度为$O(3N_qCMK)$。给定采样坐标偏移和注意力权重，式2的计算复杂度为$O(N_qC^2 + N_qKC^2+5N_qKC)$，其中$5N_qKC$中的因子5是因为注意力中双线性插值和加权和。另一方面，我们还可以在采样之前计算$\mathbf{W}_m'\mathbf{x}$，因为它与查询独立，因此式2的计算复杂度变为$O(N_qC^2 + \min(HWC^2, N_qKC^2) + 5N_qKC + 3N_qCMK)$。在唯美的实验中，默认情况下，$M=8$、$K\le4$和$C=256$，因此$5K+3MK < C$，计算复杂度为$O(2N_qC^2 + \min(HWC^2, N_qKC^2))$。

#### A.2	Constructing Multi-Scale Feature Maps for Deformable DETR

​		如4.1节和图3，从ResNet的$C_3$到$C_5$的输出特征上提取（利用$1\times1$卷积转换）编码器的输入多尺度特征图$\{\mathbf{x^l}\}_{l=1}^{L-1}(L=4)$。在最终的$C_5$阶段上使用步长为2的$3\times3$卷积获得最低分辨率的特征图$\mathbf{x}^L$。注意，没有使用FPN，因为我们提出的多尺度可变形注意力自身可以交换多尺度特征图中的信息。

![fig3](images/Deformable%20DETR/fig3.png)

#### A.3	Bounding Box Prediction in Deformable DETR

​		由于多尺度可变形注意力模块在参考点周围提取图像特征，所以我们设计检测头来将边界框预测为与参考点对应的相对偏移，从而减小优化难度。参考点用作边界框中心。检测头预测与参考点的相对偏移$\hat{\mathbf{p}}_q = (\hat{p}_{qx}, \hat{p}_{qy})$，即$\hat{\mathbf{b}}_q = \{\sigma(b_{qx} + \sigma^{-1}(\hat{p}_{qx}), \sigma(b_{qy} + \sigma^{-1}(\hat{p}_{qy})), \sigma(b_{qw}),\sigma(b_{qh})\}$，其中通检测头预测$b_{q\{x,y,w,h\}}\in\mathbb{R}$。$\sigma$和$\sigma^{-1}$分别表示sigmoid和逆sigmoid函数。$\sigma$和$\sigma^{-1}$保证$\hat{\mathbf{b}}$具有归一化坐标，即使$\hat{\mathbf{b}}_q\in[0,1]^4$。以这种方式，学习到的解码器注意力与预测边界框有强相关性，其也加速收敛。

#### A.4	更多实现细节

**Iterative Bounding Box Refinement**	这里，每个解码器层基于前一层的预测精炼边界框。假设解码器层的数量为$D$（例如$D=6$），给定由第$d-1$个解码器层预测的归一化边界框$\hat{b}_q^{d-1}$，第$d$个解码器层精炼边界框：

$$\hat{\mathbf{b}}_q^d=\{\sigma(\Delta b_{qx}^d + \sigma^{-1}(\hat{b}_{qx}^{d-1})), \sigma(\Delta b_{qy}^{d-1}+\sigma^{-1}(\hat{b}_{qy}^{d-1})),\sigma(\Delta b_{qw}+\sigma^{-1}(\hat{b}_{qw}^{d-1})), \sigma(\Delta b_{qh}^d + \sigma^{-1}(\hat{b}_{qh}^{d-1}))\},$$

其中$d\in\{1,2,\cdots,D\}$，在第$d$个解码器层预测$\Delta b_{q\{x,y,w,h\}}^d\in\mathbb{R}$。不同解码器层的预测头没有共享参数。初始边界框设置为$\hat{b}_{qx}^0 = \hat{p}_{qx}$、$\hat{b}_{qw}^0=0.1$以及$\hat{b}_{qh}^0=0.1$。系统对于$b_{qw}^0$和$b_{qh}^0$的选择很鲁棒。我们将它们设置为0.005、0.1、0.2、0.5，并获得相似的性能。为了稳定训练，与Teed&Deng（2020）相似，梯度仅反向传播通过$\Delta b_{q\{x,y,w,h\}}^d$，并在$\sigma^{-1}(\hat{b})_{q(x,y,w,h)}^{d-1}$处阻塞。

​		在迭代边界框精炼中，对于第$d$个解码器层，我们从第$d-1$个解码器层中预测的边界框$\hat{\mathbf{b}}_q^{d-1}$。对于第$d$个解码器层的cross-attention模块中的式（3），$(\hat{b}_{qx}^{d-1},\hat{b}_{qy}^{d-1})$作为新的采样点。采样点偏移$\Delta \mathbf{p}_{mlqk}$也由边界框大小调节，即$(\Delta p_{mlqkx}\hat{b}_{qw}^{d-1},\Delta p_{mlqky}\hat{b}_{qh}^{d-1})$。如此的修改式采样位置与前面预测边界框的中心和大小相关。

**Two-Stage Deformable DETR**	在第一阶段中，给定编码器的输出特征图，将检测头用于每个像素。检测头具有用于边界框回归的3层FFN，以及用于边界框二值分类（即前景和背景）的线性映射。令$i$索引具有2维归一化坐标$\hat{p}_i=(\hat{p}_{ix}, \hat{p}_{iy})\in[0,1]^2$的特征层$l_i\in\{1,2,\cdots,L\}$的一个像素，其对应的边界框预测为：

$$\hat{\mathbf{b}}_i =\{\sigma(\Delta b_{ix} + \sigma^{-1}(\hat{p}_{ix})),\sigma(\Delta b_{iy} + \sigma^{-1}(\hat{p}_{iy})),\sigma(\Delta b_{iw} + \sigma^{-1}(2^{l_i}s)),\sigma(\Delta b_{ih} + \sigma^{-1}(2^{l_i}s))\},$$

其中基目标尺度$s$设置为0.05，由边界框回归分支预测$\Delta b_{i\{x,y,w,h\}}\in\mathbb{R}$。DETR中的匈牙利损失用于训练检测头。

​		给定第一阶段的预测边界框，选择top scoring边界框作为区域提议。在第二阶段中，这些提议被馈入解码器作为迭代边界框精炼的初始边界框，其中目标查询的位置嵌入被设置为区域提议坐标的位置嵌入。

**多尺度可变形注意力的初始化**	在我们的实验中，注意力头的数量设置为$M=8$。在多尺度可变形注意力模块中，随机初始化$\mathbf{W}_m'\in\mathbf{R}^{C_v \times C}$和$\mathbf{W}_m\in\mathbb{R}^{C \times C_v}$。预测$A_{mlqk}$和$\Delta \mathbf{p}_{mlqk}$的线性映射的权重参数初始化为零。初始化线性映射的偏置参数使得$A_{mlqk} = \frac{1}{LK}$和$\{\Delta \mathbf{p}_{1lqk}=(-k,-k)$、$\mathbf{p}_{2lqk}=(-k,0)$、$\mathbf{p}_{3lqk}=(-k,k)$、$\mathbf{p}_{4lqk}=(0,-k)$、$\mathbf{p}_{5lqk}=(0,k)$、$\mathbf{p}_{6lqk}=(k,-k)$、$\mathbf{p}_{7lqk}=(k,0)$、$\mathbf{p}_{8lqk}=(k,k)\}$。

对于_iterative bounding box refinement_，初始化的解码器中的$\Delta \mathbf{p}_{mlqk}$预测的偏置参数进一步乘以$\frac{1}{2K}$，使得初始时的所有采样点位于前一解码器层预测的对应边界框中。