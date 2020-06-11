# cv_papers

### 描述

计算机视觉方面的分类、对象检测、图像分割、人脸检测、OCR等中文翻译

### Detections

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