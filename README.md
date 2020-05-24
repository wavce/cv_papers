# cv_papers

### 描述

计算机视觉方面的分类、对象检测、图像分割、人脸检测、OCR等中文翻译

### Detections

[4. Cheaper Pre-training Lunch: An Efficient Paradigm for Object Detection](./detections/Cheaper Pre-training Lunch.md)

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

[1. Object as Points (CenterNet)](./detections/CenterNet.md)

​		本文将目标检测建模为单点——目标边界框的中心点。中心点通过关键点估计获得，并同时回归目标的其他属性，例如大小、3D位置、方向甚至姿态。CenterNet在MS COCO数据集上获得最佳的速度准确率平衡，即142FPS获得28.1%的AP、52FPS获得37.4%的AP、多尺度测试下以1.4FPS的速度获得45.1%的AP。使用相同的方法估计KITTI基准测试中的3D边界框，以及COCO关键点数据集中的人体姿态。