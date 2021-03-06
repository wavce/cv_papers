# cv_papers

### 描述

计算机视觉方面的分类、对象检测、图像分割、人脸检测、OCR等中文翻译。

[27.CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows](./backbones/CSWin.md)

​		本文提出Cross-Shaped Window self-attention，其在水平和垂直条纹中进行自注意力操作，这种条纹可以让模型获得强建模能力，同时限制计算成本。根据网络深度调整条纹宽度：浅层有小宽度、深层有更大的宽度。更大的条纹宽度鼓励长范围元素之间的更强连接，并在减小计算成本的情况下获得更好的网络容量。作者还提出Locally-enhanced Positional Encoding （LePE），可以用于任意大小输入图像。代码见https://github.com/microsoft/CSWin-Transformer。

[26.Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet](./backbones/T2T.md)

​	作者发现：1）输入图像的简单令牌化难以建模相邻像素之间的局部结构，如边缘和线条，从而导致较低的训练样本效率；2）ViT 的冗余注意力主干设计导致固定计算预算和有限训练样本的有限特征丰富性。作者提出Tokens-To-Token Vision Transformer （T2T-ViT），其包含1）逐层令牌到令牌 (T2T) 转换，通过递归地将相邻令牌聚合为一个令牌（Tokens-To-Token），逐步将图像结构化为令牌，这样可以对由周围令牌表示的局部结构进行建模，并且可以减少令牌长度；2）经过实证研究后，由 CNN 架构设计驱动的视觉transformer具有深窄结构的高效主干。代码见https://github.com/yitu-opensource/T2T-ViT。

[25.LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference](./backbones/LeViT.md)

​		本文提出用于快速推理图像分类的混合神经网络，该网络将卷积和transformer结合，在不同计算平台上获得极快的推理速度和不错的分类精度。代码见https://github.com/facebookresearch/LeViT。

[24.Transformer in Transformer](./backbones/TNT.md)

​		本文向视觉transformer块中添加一个新的transformer块，从在捕获贴片级的全局信息时，还能获取像素级的局部信息。具体实现见https://github.com/huawei-noah/noah-research/tree/master/TNT。

[33. EfficientNetV2: Smaller Models and Faster Training](./backbones/EfficientNetV2.md)

​		本文针对EfficientNets的训练和推理耗时长的缺点，提出一系列改进：（1）加入Fused-MBConv的新搜索空间，（2）渐进式学习策略，将训练过程分为多个阶段，然后依次增加图像大小和加强正则化。代码和权重见https://github.com/google/automl/tree/master/efficientnetv2。

[32. Container: Context Aggregation Network](./backbones/Container.md)

​		本文证明许多流行架构的主要差异源于其聚合模块的变化。这些差异实际上可以表示为聚合器内的亲和矩阵（`或者翻译为相似矩阵`）（affinity matrix）的变体，该矩阵用于确定查询向量与其上下文之间的信息传播。例如，在ViTs[14、15]中，这种亲和矩阵使用键和查询计算动态生成；但是在Xception架构[8]（其采用深度卷积）中，亲和矩阵示静态的——不管位置如何，亲和权重都相同，并且不管大小如何，它们在所有输入图像上都相同。最后，MLP-Mixer [50] 还使用了一个静态亲和矩阵，该矩阵在输入的范围内变化。

​		沿着这种统一观点，作者提出了 CONTAINER（上下文聚合网络），这是一个用于多头上下文聚合的通用构建块。CONTAINER块包含基于聚合的静态亲以及动态亲和，其使用可学习的混合系数组合。这保证CONTAINER块处理长范围信息，同时是仍利用局部卷积操作的归纳偏置。CONTAINER块易于实现，可以很容易地替代到许多当今的神经架构中，并产生高性能的网络，同时收敛速度更快，数据效率更高。

[31.Twins: Revisiting the Design of Spatial Attention in Vision Transformers](./backbones/Twins.md)

​		本文回顾了视觉Transformer中的空间注意力的设计。首先发现PVT中的全局子采样注意力高度有效，并利用合适的位置编码，可以产生与Swin相当的性能，由此产生了Twins-PCPVT。进一步，作者提出一种简单的空间注意力替换PVT中缺乏有效全局子采样的注意力，即受可分离卷积启发的可分离注意力机制——spatially separable self-attention（SSSA），SSSA包含两种操作（i）locally-grouped self-attention（LSA）和（ii）global sub-sampled attention（GSA），其中LSA捕获细粒度和短距离信息，GSA处理长距离和全局信息。代码见https://git.io/Twins。

[30.Dual-stream Network for Visual Recognition](./backbones/DSNet.md)

​		本文提出一种通用的Dual-stream Network（DS-Net）来充分利用局部和全局特征的表示能力进行图像分类。DS-Net可以同时计算细粒度和集成特征，并高效地融合它们。具体而言，用一种Intra-scale Propagation模块处理每个块中的不同分辨率，以及一种Inter-Scale Alignment模块来进行两个尺度上的跨特征信息交互。此外，还设计一种Dual-strem FPN（DS-FPN）来进一步增强下游密集预测的上下文信息。

[29.ResT: An Efficient Transformer for Visual Recognition](./backbones/ResT.md)

​		本文提出一种高效的多尺度视觉Transformer（称为ResT），其能够作为图像识别的通用主干。与已有的Transformer方法不同，已有的方法利用标准的Transformer块来以固定分辨率处理原始图像，ResT有几个优势：（1）构建内存高效的多头自注意力，其通过简单的depth-wise卷积压缩内存，并在保持多头的多样性能力的同时，跨注意力头维度映射交互；（2）将位置编码构建为空间注意力，其更灵活，并且可以处理任意大小的输入图像，而没有插值或微调；（3）与其在每个阶段开始时直接标记化，将补丁嵌入设计为在令牌映射上具有步幅的堆叠的重叠卷积操作。在图像分类和下游任务上全面验证了 ResT。实验结果表明，所提出的ResT可以大幅领先最近的SOTA骨干，这表明ResT的有作为强骨干的潜力。代码和模型见https://github.com/wofmanaf/ResT。

[28.SOLOv2: Dynamic, Faster and Stronger](./instance_segmentations/SOLOv2.md)

​		本文在SOLO的基础之上提出将掩膜分支解藕为掩膜核分支和掩膜特征分支，用预测的掩膜核卷积掩膜特征。同时，还提出Matrix NMS，加速NMS过程，同时也相比传统NMS取得准确率上的提高。代码见https://github.com/aim-uofa/AdelaiDet。

[27. GhostNet: More Features from Cheap Operations](./backbones/GhostNet.md)

​		本文提出一种新颖的Ghost模块来从便宜的操作（cheap operations）中生成更多特征图。基于一组固有的特征图，我们使用一系列便宜成本（cheap cost）的线性变换来生成许多ghost特征图，这些特征图可以完整反映固有特征潜在的信息。所提出的Ghost模块可以作为即插即用的组件来更新已有的卷积神经网络。设计Ghost块以堆叠成Ghost模块，然后建立轻量化的GhostNet。在基准测试上的实验证明，所提出的Ghost模块是基准模型中卷积层的令人印象深刻的替代方案，在ImageNet ILSVRC-2012分类数据集上，GhostNet以相似的计算成本获得比MobileNetV3更高的识别性能（例如75.7%的top-1准确率）。代码见https://github.com/iamhankai/ghostnet。

[26.Activate or Not: Learning Customized Activation](./backbones/ACON.md)

​		本文提出一种简单、高效且通用的激活函数（称为ACON），其通过数据学习激活神经元与否。作者将通用的Maxout系列解释为ACON系列，特别地，作者任务Swish是ReLU的平滑近似，这与ACON是其他Maxout系列的近似相似。ACON在轻量化网络和更大的网络上都获得较大的性能改进，可以在https://github.com/nmaac/acon上找到代码。

[25. Rethinking Channel Dimensions for Efficient Model Design](./backbones/ReXNet.md)

​		在有限的计算成本中设计高效的模型是具有挑战的。作者认为轻量化模型的准确率收到设计习惯的局限：通道维度逐阶段的配置，看起来像网络阶段的分段线性函数。本文研究一种有效的通道尺寸配置，以实现比传统的更好的性能。为此，作者经验上研究如何通过分析输出特征的阶来恰当地设计单个层。然后，通过在计算成本限制下搜索有关通道配置的网络架构来研究模型的通道配置。基于这种研究，提出一种简单而有效的通道配置，其可以由层索引参数化。因此，在通道参数化之后提出的模型在ImageNet分类和迁移学习任务（包括COCO目标检测，COCO实例分割和细粒度分类）上均取得了卓越的性能。可通过https://github.com/clovaai/rexnet获得代码和ImageNet预训练的模型。

[24. SOLO: Segmenting Objects by Locations](./instance_segmentations/SOLO.md)

​		本文提出一种新的、简单的方法来进行实例分割。本文根据实例位置和大小将类别分配实例中的每个像素，因此很好的将实例掩膜分割转换为分类可解决的问题。将实例分割分解为语义类别分类和二值掩膜分类任务，作者完全消除了边界框预测和分组后处理过程。在准确率方面达到Mask R-CNN的水平。

[23.Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](./backbones/SwinTransformer.md)

​		本文提出一种新的视觉Transformer（称为Swin Transformer），其可以用作计算机视觉的通用主干。作者首先引入滑动窗口机制的多头自注意力，这极大地减小计算复杂度，使Transformer的计算复杂度从图像大小的二次方减小到图像大小的线性复杂度（$4hwC^2 + 2(hw)^2C \rarr 4hwC^2 + 2M^2hwC$），由于滑动窗口是以不重叠的方式均匀划分图像，这限制了它的建模能力，因此作者引入shifted window机制，即窗口配置通过将窗口从规则划分的窗口中平移$(\lfloor \frac{M}{2}\rfloor, \lfloor \frac{M}{2}\rfloor)$像素，在计算多头注意力之后，再将移动的窗口移会原来的位置。最后模型在分类、检测、实例分割和语义分割上都获得SOTA。该代码和模型将在https://github.com/microsoft/Swin-Transformer上公开提供。

[22.Revisiting ResNets: Improved Training and Scaling Strategies](./backbones/ResNet-RS.md)

​		本文认为最佳性能的缩放策略取决于训练方案，并提供了两种新的缩放策略：（1）在可能发生过拟合的方案中缩放模型深度（否则宽度缩放更可取）； （2）提高图像分辨率的速度比以前建议的要慢（Tan＆Le，2019）。使用改进的训练和缩放策略，本文设计一族ResNet架构（ResNet-RS），其在TPU上比EffcientNets块1.7倍到2.7倍，同时在ImageNet上获得相似的准确率。在大尺度半监督学习设置中，ResNet-RS获得86.2%的top-1 ImageNet准确率，同时比EfficientNet-NoisyStudent块4.7倍。训练技术提高一系列下游任务（一流的SOTA自监督算法）的性能，并扩展到Kinetics-400的视频分类上。本文建议从业者使用这些简单的经过修订的ResNets作为将来研究的基准。代码见 https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs。

[21.ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network](./backbones/ReXNet.md)

​		本文认为常规网络存在表示瓶颈，表示瓶颈会导致模型性能下降。本文分析上千个随机模型生成的矩阵秩，从而得出结论特征的矩阵秩与模型的表示瓶颈相关，秩越大模型表示能力越强。所以，本文从输入的通达数量、扩张层的数量和非线性函数（使用Swish-1替换其他非线性函数）使用三个方面来增大特征的矩阵秩。本文给出了一种新的模型设计思路。代码和预训练模型可在https://github.com/clovaai/rexnet上获得。

[20.CvT: Introducing Convolutions to Vision Transformers](./classifications/Cvt.md)

​		本文提出一种新的架构，称为Convolutional vision Transformer（CvT），其通过将卷积引入ViT中以产生两种设计的最佳效果，从而提高了Vision Transformer（ViT）的性能和效率。这通过两种主要修改来完成：包含新卷积令牌（token）嵌入的Transformer层次结构，以及利用卷积投影的卷积Transformer块。这些改变将卷积神经网络的理想特性（即平移、尺度和畸变的不变性）引入ViT架构，同时保留Transformer的优点（即动态注意力、全局上下文和更好的泛化能力）。本文通过进行广泛的实验来验证CvT，表明该方法与ImageNet-1k上的其他Vision Transformer和ResNet相比，具有更少的参数和更低的FLOP，可实现最新的性能。此外，当在更大的数据集（例如ImageNet-22K）上预训练，并在下游任务上微调时，仍能维持性能增益。在ImageNet-22k上预训练，本文的CvT-W24在ImageNet-1K验证集上获得87.7%的top-1准确率。最后，本文的结果表明位置嵌入编码（已有的Vision Transformer中的关键组件）可以在本文的模型中被安全移除，其简化更高分辨率视觉任务的设计。代码见https://github.com/leoxiaobin/CvT。

[19.Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](./detections/PVT.md)

​		本文开发了一种具有金字塔结构的Transformer网络，其可以用于各种密集预测任务（目标检测、实例分割和语义分割）。代码见https://github.com/whai362/PVT。

[18. You Only Look One-level Feature](./detections/YOLOF.md)

​		本文提出一种没有FPN的一阶段检测器，其指出是分治法解决了目标检测中的优化问题，而不是多尺度特征融合。这种方法称为YOLOF，YOLOF包含两种关键组件：Dilated Encoder和Uniform Matching，并带来与RetinaNet相当的性能，但比RetinaNet快2.5倍。YOLOF还可以匹配DETR的性能，训练速度快了7倍。代码见https://github.com/megvii-model/YOLOF。

[17. Probabilistic two-stage detection](./detections/CenterNet2.md)

​		本文开发了两阶段目标检测的概率解释。第一阶段应当推断恰当的object-vs-background似然，然后，其应当告知检测器的整体分数。标准区域提议网络（RPN）不能很好地推断出这种可似然，但是许多一阶段检测器可以。介绍如何构从任意的SOTA一阶段检测器构建概率两阶段检测器。产生的检测器比它们一阶段和两阶段先驱都快而准。本文的检测器在COCO test-dev上，利用单尺度测试下获得56.4 mAP，比所有公开结果都好。使用轻量的骨干，在Titan Xp上，本文的检测器在COCO上以33fps获得49.2 mAP，比流行的YOLOv4模型好。代码和模型见https://github.com/xingyizhou/CenterNet2。

[16. VarifocalNet: An IoU-aware Dense Object Detector](./detections/VarifocalNet.md)

​		本文提出学习IoU-aware classification score（IACS），其将预测的IoU作为分类学习目标，使得它能够同时表示目标存在的置信度和定位准确率，从而在密集目标检测器中产生更准确的检测排序，更有利于NMS。并且有别于GFL：（1）提出VarifocalLoss，对正负样本进行非堆成加权，（2）提出星形边界框表示和边界框精炼模块。具有Res2Net-101-DCN的最佳模型在COCO test-dev上达到51.3AP，代码见代码见https://github.com/hyz-xmaster/VarifocalNet。

[15.Deformable DETR: Deformable Transformers for End-to-End Object Detection](./detections/Deformable DETR.md)

​		由于Transformer注意力模块在处理图像时存在收敛慢、特征图分辨率有限（小目标检测性能低）等问题。作者受可变形卷积的启发，提出Deformable Transformer attention module，该模块的引入加速模型收敛。其次，作者还利用多尺度特征图，进一步提高检测性能。代码见https://github.com/fundamentalvision/Deformable-DETR。

[14.Sparse R-CNN: End-to-End Object Detection with Learnable Proposals](./detections/SparseRCNN.md)

​		本文提出Sparse R-CNN，它是用于图像中目标检测的纯稀疏方法。目标检测上的已有工作严重依赖密集目标候选，例如在大小为$H\times W$的图像特征图网格上预定义的$k$个锚。然而，在我们的方法中，目标识别头提供学习的目标提议的稀疏集（共$N$个提议）来进行分类和定位。通过将$HWk$（多达数十万个）手工设计的候选目标消除为$N$个（例如100个）可学习的提议，Sparse R-CNN完全避免了与候选目标设计和多对一标签分配相关的所有工作。更重要的是，直接输出最终的预测，而没有非极大值抑制过程。Sparse R-CNN与具有挑战性的COCO数据集上公认的检测器基准具有同等的准确性、运行时间和训练收敛性能，例如使用ResNet-50 FPN模型，在标准的3倍训练调度下，以22FPS的速度获得44.5AP。代码见https://github.com/PeizeSun/SparseR-CNN。

#### [13. End-to-End Object Detection with Fully Convolution Network](./detections/DeFCN.md)

​		本文为了消除NMS，提出一种新的标签分配方案，该方案称为Prediction-aware One-To-One（POTO），同时，引入3D Max Filtering来抑制跨FPN尺度之间的重复预测。为了获取更辨别性特征（一对一标签分配提供更少的监督，使网络难以学习强壮而鲁棒的特征表示），作者还引入辅助损失，该损失与ATSS相似。详细代码见 https://github.com/Megvii-BaseDetection/DeFCN。

#### [12. OneNet: Towards End-to-End One-Stage Object Detection](./detections/OneNet.md)

​		本文将分类成本（分类损失）和位置成本（定位损失）引入到标签分配中，从而形成成为Minimum Cost Assignment标签分配方案，即将分类成本和定位成本之和最小的样本分配给ground-truth目标。这一简单而有效分配策略之后，消除了一阶段密集检测器中的NMS，从而使检测器成为真正的端到端检测器。代码见https://github.com/PeizeSun/OneNet。

#### [11.RepPoints v2: Verification Meets Regression for Object Detection](./detections/RepPointsV2.md)

​		验证和回归是神经网络中用于预测的两种通用方法。每一个都它们自己的优势：验证可以更容易准确地推断出来，而回归更高效、更适合连续目标变量。因此，仔细组合它们以利用它们的好处通常是有益的。将验证任务引入RepPoints的位置预测中，从而生成RepPoints v2，它使用不同的主干和训练方法，在COCO对象检测基准上比原始RepPoints一致地提高了约2.0 mAP。RepPoints v2还可以通过单个模型在COCO test-dev 上实现 52.1 mAP。

​		具体做法是增加一个验证分支，该分支预测角点热图和within-box foreground；然后与RepPoints进行联合推理。

#### [10.RepPoints: Point Set Representation for Object Detection](./detections/RepPoints.md)

​		现代目标检测器严重依赖矩形边界框，例如锚、提议和最终预测，来表示不同识别阶段的目标。边界框便于使用，但是仅提供目标的粗糙位置，并导致相对粗糙的目标特征提取。本文中，本文提出**RepPoints**（representative points），一种新的更精细的目标表示形式，即一组对定位和识别都有用的采样点。给定用于训练的ground-truth位置和识别目标，RepPoints学会以限制目标空间范围并指示语义上重要的局部区域的方式自动安排它自己。此外，它们不需要使用锚来采样边界框的空间。本文展示了基于RepPoints的无锚目标检测器可以与基于锚的最新检测方法一样有效，使用ResNet-101的模型在COCO test-dev 检测基准上获得46.5 AP和67.4 $AP_{50}$ 。可从https://github.com/microsoft/RepPoints获得代码。

#### [9.Probabilistic Anchor Assignment with IoU Prediction for Object Detection](./detections/PAA.md)

​		本文提出一种新颖的锚分配策略（称为Probabilistic Anchor Assignment），其自适应地将锚分为正负样本。具体做法是：首先计算以模型为条件的锚的分数，并为这些分数拟合概率分布。然后，根据锚的概率将模型分为正样本和负样本，对模型进行训练。作者还研究训练和测试目标之间的差距，并提出预测检测边界框的IoU作为定位质量的衡量，从而减小差异。分类和定位质量的组合得分作为NMS中的边界框选择度量，与所提出的锚分配策略很好对齐（这解决了训练和测试中使用的指标不对齐的问题），并产生明显的性能改进。这种分配方案可以灵活用于所有单阶段检测方法，仅需添加一个卷积层。

#### [8.Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection](./detections/GFL.md)

​		本文深入研究了以上三个基本元素的表示形式：质量估计、分类和定位。在现有的实践中发现两个问题，包括（1）训练和推理中质量估计和分类的使用的不一致（即单独训练，但是在测试时，复合使用），以及（2）当存在歧义和不确定性时（在复杂场景中通常是这种情况），定位的inflexible Dirac Delta分布。为了处理这个问题，作者为这些元素设计新的表示。具体而言，作者将质量估计融入分类预测向量以构成定位质量和分类的联合表示，并使用一个向量来表示边界框定位任意分布。改进的表示消除不一致风险，并准确描述真实数据中的灵活分布，但是对于包含连续标签，其超出Focal Loss的范畴。然后，作者提出Generalized Focal Loss（GFL），将Focal Loss从离散形式推广到连续版本，以实现成功的优化。在COCO test-dev中，GFL使用ResNet-101骨干网达到45.0％的AP，在相同的骨干网络和训练设置下，以更高或相当的推理速度超过最先进的SAPD（43.5％）和ATSS（43.6％） 。值得注意的是，本文的最佳模型可以在单个2080Ti GPU上以10 FPS的速度实现48.2％的单模型单尺度AP。 代码和预训练模型可在https://github.com/implus/GFocal获得。

#### [7. AutoAssign: Differentiable Label Assignment for Dense Object Detection](./detections/AutoAssign.md)

​		本文提出了一种具有完全可微标签分配策略的无锚目标检测器，称为AutoAssign。它通过生成正负权重图来自动确定正/负样本，以动态修改每个位置的预测值。具体而言，我们提出中心加权模块（center weighting module）来调整类特定的先验分布（category-specific prior distribution），以及置信度加权模块（confidence weighting module）以适应每个实例的特定分配策略。整个标签分配是可微的，并不需要额外的修改便可迁移到不同数据集和任务。MS COCO上的大量实验证明，我们的方法稳定地超过其他最佳的采样策略大约1% $AP$。此外，我们的最佳模型获得52.1%的 $AP$，这比所有已有一阶段检测器好。此外，在其数据集（例如PASCAL VOC、Object365和WiderFace）上的实验也说明AutoAssign的广泛应用能力。

#### [6.DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution](./detections/DetectoRS.md)

​		许多现代目标检测器通过使用两次观察和思考（looking and thinking twice）的机制，表现出出色的性能。本文在backbone中探索这种机制已进行目标检测（即多次提取图像特征）。在宏观水平，本文提出递归特征金字塔（Recursive Feature Pyramid），它结合从特征金字塔网络到自下而上的骨干层的额外反馈连接。在微观水平，本文提出Switchable Atrous Convolution，其利用不同的atrous rate卷积特征，并聚合switch函数聚合结果。结合它们产生DetectoRS，其显著提高目标检测的性能。在COCO test-dev上，DetectoRS获得用于目标检测的54.7％的边界框AP，用于实例分割的47.1％的掩膜AP和用于全景分割的49.6％的PQ。代码见 https://github.com/joe-siyuan-qiao/DetectoRS 。

#### [5. SpineNet:  Learning Scale-Permuted Backbone for Recognition and Locatlization](./detections/SpineNet.md)

​		本文通过NAS搜索用于目标检测的新backbone，这种backbone不再用于如ResNet那种递减的特征图。通过搜索得到的backbone具有scale-permuted的中间特征和cross-scale connection。作者将这种架构称为SpineNet，它在COCO上的性能比ResNet-FpN高3%，而单模型SpineNet-190的AP为52.1%。同时，SpineNet在iNaturalist细粒度数据集上的top-1精度也提高5%。代码见 https://github.com/tensorflow/tpu/tree/master/models/official/detection 。

#### [4. Cheaper Pre-training Lunch: An Efficient Paradigm for Object Detection](./detections/Cheaper Pre-training Lunch.md)

​		本文提出一种新的主干网络预训练范式——Jigsaw预训练，这种预训练仅需要检测数据集，它从图像上裁剪出包含目标的patch样本，然后将其中的4个样本拼接为一幅完整图像作为训练样本。这种方式的预训练消除了额外数据集的负担，节约整体训练时间（与ImageNet预训练相比，节约了1/4的时间）。同时，为了更好的预训练，还提出了有效感受野自适应的损失函数。MS COCO上的大量实验表明，该方法能够实现同等甚至更好的性能。

#### [3. Scale-Equalizing Pyramid Convolution for Object Detection](./detections/SEPC.md)

​		本文提出Pyramid Convolution（PConv），它是一种改进的3D卷积，并用于提取新的特征金字塔。朴素的金字塔卷积以及RetinaNet head的设计实际上最适用于从高斯金字塔中提取特征，而高斯金字塔的特性很难被特征金字塔所满足。为了缓解这种差异，作者构建scale-equalizing pyramid convolution（SEPC），其仅在高级特征图上对齐共享金字塔卷积核。PConv的输出为$y^l = \mbox{Upsample}(w_1 \ast x^{l+1}) + w_0 \ast x^l + w_{-1}\ast_{s2} x^{l-1} $，其中$w$为卷积，本文中使用DeformableConv2D。作者还提取iBN用于收集所有金字塔层的统计量。作者还提出SEPC-Lite，其在P3上使用常规Conv2D。

​		主要贡献如下：

​		（1）提出轻量的金字塔卷积（PConv），以在特征金字塔内部进行3-D卷积，从而满足尺度间的相关性。

​		（2）开发尺度均衡特征金字塔（scale-equalizing pyramid convolution：SEPC），以通过仅在高级特征图上对齐共享的PConv核来缓解特征金字塔和高斯金字塔之间的差异。

​		（3）该模块以可忽略的推理速度提高了性能（在最先进的单阶段目标检测器上提高了3.5AP）。

​		代码已开源：https://github.com/jshilong/SEPC 。

#### [2. Revisiting the Sibling Head in Object Detector（TSD）](./detections/TSD.md)

​		本文的研究启发是，基于锚的目标检测其中的sibling head中分类和回归特征不匹配（或者说不对齐，或者说存在空间上的纠缠），即分类效果好的特征不一定定位准确，而定位准确的特征分类效果会差。虽然Double-Head R-CNN能够一定程度上解耦分类和定位空间上的纠缠，但实际上它仅仅是减少了两个任务的共享参数，而输入Double Head两个分支的提议仍由一个RoI池化产生，所以两个任务之间仍未能完全解耦。因此，作者重新审视这一问题，提出任务感知的空间解耦，本文作出的了如下贡献：
​		1）深入研究基于RoI的检测器中复杂任务背后的主要障碍，并揭示了限制检测性能上限的瓶颈。
​		2）提出简单的操作（称为task-aware spatial disentanglement（TSD））来处理复杂的任务冲突。通过任务感知的提议估计和检测头部，可以生成任务特定的特征表示来消除分类和检测之间的折衷。
​		3）进一步提出 progressive constraint（PC）来扩大TSD和经典sibling head之间的性能边界。
​		4）该方法使用单模型的ResNet-101骨干获得49.4的mAP，而使用SENet154的模型获得51.2的mAP。
​		代码已开源：https://github.com/Sense-X/TSD 。

#### [1. Object as Points (CenterNet)](./detections/CenterNet.md)

​		本文将目标检测建模为单点——目标边界框的中心点。中心点通过关键点估计获得，并同时回归目标的其他属性，例如大小、3D位置、方向甚至姿态。CenterNet在MS COCO数据集上获得最佳的速度准确率平衡，即142FPS获得28.1%的AP、52FPS获得37.4%的AP、多尺度测试下以1.4FPS的速度获得45.1%的AP。使用相同的方法估计KITTI基准测试中的3D边界框，以及COCO关键点数据集中的人体姿态。

### OCR

#### [1. An End-to-End TextSpotter with Explicit Alignment and Attention](./OCR/TextSpotter.md)

​		自然场景中的文本检测和识别长期被认为是两个依次处理的单独任务。由于学习困难和收敛速度的显着差异，共同训练两项任务并非易事。在这项工作中，作者提出一种概念上简单而高效的框架，其在统一框架中同时处理这两个任务。主要贡献有三个方面：（1）提出一种新颖的文本对齐（text-alignment）层，其运行它精准地计算任意方向文本实例的卷积特征，这是提升性能的关键；（2）将字符空间信息作为显式监督，从而引入字符注意力机制（character attention mechanism），这在识别上产生很大改进。两种技术以及用于单词识别的新RNN分支无缝集成到了一个可端到端训练的单一模型中。这使两个任务可以通过共享卷积特征来协同工作，这对于识别具有挑战性的文本实例至关重要。

#### [2. What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis](./OCR/WhatIsWrongInSTR.md)

​		本文贡献如下：第一，检查训练和评估数据集的不一致之处，以及由于不一致导致的性能差距。第二，引入一个统一的四阶段STR框架，大多数现有的STR模型都适合该框架。使用此框架可对先前提出的STR模块进行广泛评估，并发现先前未探索的模块组合。第三，在一组一致的训练和评估数据集下分析模块对性能、准确性、速度和内存需求的贡献。

### Classifications

#### [1. Designing Network Design Spaces](./classifications/regnet.md)

​		本文提出一种新的网络设计范式。本文的目标是帮助增进对网络设计的理解，并发现可在各种设置之间泛化的设计原则。本文不着重于设计单个网络实例，而是设计可参数化网络总体的网络设计空间。整个过程类似于经典的网络手动设计，但提升到设计空间级别。使用本文的方法，探索网络设计的结构方面，并得出一个低维设计空间，该空间由简单的常规网络组成，我们称之为RegNet。RegNet参数化的核心见解非常简单：良好网络的宽度和深度可以用量化的线性函数来解释。本文分析RegNet的设计空间，并得出与当前网络设计实践不符的有趣发现。RegNet设计空间提供简单而快速的网络，这些网络可以在各种各样的FLOP下正常工作。在可比的训练设置和FLOP之下，RegNet模型优于流行的EfficientNet模型，而在GPU上的速度提高5倍。代码见https://github.com/facebookresearch/pycls。

### 长尾识别

#### [1.Class-Balanced Loss Based on Effective Number of Samples](./long_tail/class_balanced_loss.md)

​		本文旨在解决长尾数据问题，提出一种新颖的理论框架，其通过有效样本量来对损失函数重加权。样本的有效数量定义为样本的体量，并可以通过简单的公式$(1-\beta^n)/(1-\beta)$计算，其中样本的数量和$\beta \in [0, 1)$为超参数。在人工诱导的长尾CIFAR数据集和包括ImageNet和iNaturalist的大规模数据集上进行了综合实验。实验结果证明，当利用所提出的类平衡损失训练时，网络能够在长尾数据集上获得明显的性能增益。代码见 https://github.com/richardaecn/class-balanced-loss。

#### [2. Equalization Loss for Long-Tailed Object Recognition](./long_tail/EQL.md)

​		本文从一个新颖的角度分析这个问题：一个类的每个正样本可以看作其他类的负样本，使得尾部类别接收到更多令人沮丧的梯度。在此基础上，本文提出一个简单而有效的损失（称为均衡损失（equalization loss））通过简单地忽略稀有类别的梯度来解决长尾稀有类别的问题。均衡损失保护稀有类别的学习在网络参数更新过程中处于不利地位。因此，该模型能够更好地学习稀有类目标的判别特征。没有任何花里胡哨的操作，在LVIS基准测试的稀有和普遍类上，与Mask R-CNN相比，本文的方法分别获得4.1%和4.8%的性能增益。利用高效的均衡损失，在LVIS Challenge 2019上获得第一名。代码见 https://github.com/tztztztztz/eql.detectron2 。

### OCR

#### [2.TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes](./OCR/TextSnake.md)

​		本文提出一种新颖的文本表示方法——SnakeText，其目的是解决轴对齐矩形、旋转矩形或四边形表示的缺陷。TextSnake中，文本实例被描述为以对称轴为中心的有序、重叠的圆序列，每个圆都与潜在的可变半径和方向相关。具体而言，模型会预测文本区域、文本中心线和几何图，几何图包含圆的半径、$\cos\theta$和$\sin\theta$（细节见3.6节的标签生成）。	基于TextSnake的文本检测器在Total text和SCUT-CTW1500这两个最新发布的基准测试（特别是自然图像中的曲线文本）以及广泛使用的数据集ICDAR 2015和MSRA-TD500上达到了最先进或可比的性能。具体而言，在Total-Text上，TextSnake的F-measure比基线高40%。

#### [1.Look More Than Once: An Accurate Detector for Text of Arbitrary Shapes](./OCR/LOMO.md)

​		受CNN感受野以及诸如采用矩形边界框或四边形的简单表示来描述文本的限制，因此，在处理更具挑战行的文本实例（例如极长的文本和任意形状的文本）时，过去的工作可能存在短板。为了处理这两个问题，本文提出新的文本检测器（成为LOMO），其多次定位文（或者说LOok More than Once）。LOMO包含direct regressor（DR）、iterative refinement module（IRM）和shape expression module（SEM）。

​		首先，通过DR分支生成四边形格式的文本提议。该模块借鉴EAST的架构，但将骨干替换为Resnet50-FPN。

​		然后，IRM通过基于初步提议中提取的特征块的迭代式地精炼逐步感知整个长文本。IRM首先通过RoI transform layer获取$1 \times 8 \times 64 \times 128$特征块；然后将提取到的特征馈入3个$3 \times 3$卷积层；接下来引入角注意力机制，它使用$1 \times 1$和sigmoid获得4个角的注意力图；最后通过group dot production和sum reduction operation获得4个角回归特征。

​		最后，引入SEM来结合文本实例的几何特征（包括文本区域、文本中心线和边界偏移等）以重构出更精确的不规则文本表示。SEM回归这三种类型的特征，从而重构文本实例的精准形状表示。

​		在几种公开基准测试（包括ICDAR2018-RCTW、SCUT-CTW1500、Total-Text、ICDAR2015和ICDAR17-MLT）上的最佳结果证实LOMO惊人的健壮性和有效性。

### MOTs

#### [4. A Simple Baseline for Multi-Object Tracking](./mots/FairMOT.md)

​		本文中研究将检测和重识别结合后，性能衰退的关键原因，并根据提出的简单基线来处理这个问题。作者分析认为影响跟踪性能的三个关键因素：（1）anchor不适合ReID，因为首先，可能存在多个anchor负责估计相同的目标；其次，8倍的下采样对于ReID太粗糙，目标中心和用于预测目标ID的粗糙anchor位置提取的特不能对齐。（2）Multi-layer Feature Aggregation，因为ReID需要地低级特征和高级特征来促进小型和大型目标。（3）Dimensionality of the ReID Features，ReID一般需要高维特征，因为MOT的训练图像比ReID少，因此使用较低维特征有利于MOT。本文的算法借鉴CenterNet的思想，对于Identity Embedding Loss，使用分类损失，计算softmax 损失。

#### [3. RetinaTrack: Online Single Stage Joint Detection and Tracking](./mots/RetinaTrack.md)

​		本文提出一种简单而有效的检测与跟踪的联合模型，称为RetinaTrack。其基础为RetinaNet，网络输出后FPN预测分为两个阶段：与RetinaNet相同的Task-Shared Post-FPN层和Task-Specific Post-FPN层。在Task-Specific Post-FPN层中，每个anchor都有一个独立的卷积层即将原有输出为$A * K$卷积（对于分类，$K$为类别数；对于回归，$K=4$）分解为$A$个输出为$K$的卷积，这样的操作有效被遮挡的目标与遮挡目标的中心重合的问题。Task-Specific Post-FPN层有三个分支：分类、边界框回归和嵌入，其学习到的是Per-anchor instance-level特征。对于嵌入，使用具有BatchHard策略的triplet loss。

#### [2. Tracking without bells and whistles](./mots/Tracktor.md)

​		本文提出的跟踪器（Tracktor）没有在数据上进行训练或优化。Tracktor利用目标检测器的边界框回归来预测目标在下一帧中的位置，即，首先，目标检测器的回归将帧$t-1$的已有的跟踪边界框$b_{t-1}^k$对齐到帧$t$中的目标的新位置。然后，使用新边界框位置对应的目标分类得分$s_t^k$来停用潜在遮挡的跟踪。其次，如果检测与激活跟踪$B_t = \{b_t^{k_1},b_t^{k_2},\cdots\}$没有实质上的IoU，那么初始化新的跟踪。

#### [1. Simple Online and Realtime Tracking with a Deep Association Metric](./mots/DeepSort.md)

​		在Sort跟踪算法的基础之上增加外观特征，即引入ReID，便形成DeepSort。

### SuperResolution

#### [1. Learning a Deep Convolutional Network for Image Super-Resolution](./SuperResolution/SRCNN.md)

​		本文提出用于单幅图像超分辨率（Super Resolution：SR）的深度学习方法。本文的方法直接学习低/高分辨率图像之间的端到端映射。这种映射表示为深度卷积神经网络，该网络采用低分辨率图像作为输入，并输出高分辨率图像。作者进一步证明传统基于稀疏编码的SR方法以可以视为深度卷积网络。但是，与单独处理每个组件的传统方法不同，本文的方法联合优化所有层。本文的深度CNN具有轻量化结构，同时展现了最先进的恢复质量，并为实际在线使用提供快速的速度。

#### [2. Deeply-Recursive Convolutional Network for Image Super-Resolution](./SuperResolution/DRCN.md)

​		本文提出了一种使用深度递归卷积网络（DRCN）的图像超分辨率方法（SR）。该网络具有非常深的递归层（最多16个递归）。递归深度的增加可以提高性能，而无需为其他卷积引入新参数。尽管有优势，但由于存在梯度爆炸/消失，因此使用标准梯度下降方法学习DRCN非常困难。为了减轻训练的难度，作者提出了两个扩展：递归监督和跳过连接（skip-connection）。 

#### [3.Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](./SuperResolution/ESPCN.md)

​		基于深度神经网络的几种模型在单图像超分辨率的重建准确率和计算性能方面都取得巨大成功。在这些方法中，在重建之前，使用单个滤波器将低分辨率（Low Resolution：LR）输入图像放大到高分辨率（High Resolution：HR）空间，通常使用的是bicubic插值。这意味着SR操作在HR空间完成。我们证明这是次优的，并增加了计算复杂度。本文提出了第一个能够在单个K2 GPU上对1080p视频进行实时SR的卷积神经网络（CNN）。为了实现这一点，作者提出了一种新颖的CNN架构，其中在LR空间中提取了特征图。此外，引入高效亚像素卷积层（efficient sub-pixel convolution layer），该层学习一系列的放大滤波器，以将最终的LR特征图升放大HR输出中。使用更复杂的针对每个特征图训练的更复杂的放大滤波器（upscaling filters）有效地替换了SR流水线中的手工双三次（bicubic）滤波器，同时还降低了整个SR操作的计算复杂性。使用公开数据集上的图像和视频评估我们所提出的方法，并表明该方法的效果明显更好（图像上+0.15dB，视频上+0.39dB），比以前的基于CNN的方法快一个数量级。

#### [Accelerating the Super-Resolution Convolutional Neural Network](./SuperResolution/FSRCNN.md)

​		随着深度学习用于超分辨率（SR）的成功，Super-Resolution Convolutional Neural Network（SRCNN）在速度和恢复质量方面都表现出比先前手工特征模型更优越。但是，高计算成本仍妨碍其实际要求实时性能的（4fps）应用。本文旨在加速当前的SRCNN，并提出沙漏形状的CNN结构进行更快、更好的SR。作者主要从三个方面重新设计SRCNN结构。第一，在网络的末端加入反卷积层，然后直接从低分辨率图像（没有插值）到高分辨率图像之间学习映射。第二，通过在映射之前缩小输入特征的尺寸并在之后再扩展，来重新构造映射层。第三，采用较小的滤波器尺寸，但采用更多的映射层。 所提出的模型可实现40倍以上的速度，甚至具有卓越的恢复质量。 此外，还介绍了可以在通用CPU上实现实时性能并同时保持良好性能的参数设置。 还提出了一种相应的迁移策略，用于跨不同的放大因子进行快速训练和测试。

#### [Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution](./SuperResolution/LapSRN.md)

​		本文提出Laplacian Pyramid Super-Resolution Network（LapSRN），以逐步重建高分辨率图像的子带残差（sub-band residual）。在每个金字塔层（level）中，模型采用粗糙分辨率的特征图作为输入预测高频残差，并使用转置卷积上采样到更精细的级别。该方法不需要bicubic插值作为预处理步骤，并因此极大地减小计算复杂度。作者还使用鲁棒的Charbonnier损失函数在深度监督下训练提出的LapSRN，并实现高质量的重建。此外，网络通过渐进式重建在一次前馈中生成多尺度预测，从而促进了资源感知型应用。对基准数据集的大量定量和定性评估表明，在速度和准确性方面，所提出的算法在性能方面优于最新方法。

#### [Image Super-Resolution via Deep Recursive Residual Network](./SuperResolution/DRRN.md)

​		基于CNN的模型的单图像超分辨率（Single Image Super-Resolution：SISR），由于深度网络的强大，这些CNN模型学习从低分辨率输入图像到高分辨率目标图像的有效非线性映射，但需要大量参数。本文提出非常深的CNN模型（增加到52个卷积层），称为Deep Recursive Residual Network（DRRN），其致力于建立深度而简洁的网络。具体而言，采用残差学习，以全局和局部的方式来缓解训练非常深度网络的难度；递归学习用于控制模型参数，同时增加深度。大量基准评估证明DRRN明显优于SISR中最先进的方法，同时参数量远少于这些方法。代码见https://github.com/tyshiwo/DRRN_CVPR17。

