
#### 无监督
1. Unpaired Point Cloud Completion on Real Scans Using Adversarial Training (ICML 2020)[[paper]](https://openreview.net/forum?id=HkgrZ0EYwB) [[code]](https://github.com/xuelin-chen/pcl2pcl-gan-pub)


Problem: 现有的方法都是全监督学习，依赖 残缺 - 完整 配对的训练数据，无法在真实场景（缺少对应完整点云）的场景下适用。


2. Multimodal Shape Completion via Conditional Generative Adversarial Networks (ECCV 2020)[[paper]](https://arxiv.org/abs/2003.07717)[[code]](https://github.com/ChrisWu1997/Multimodal-Shape-Completion)


Problem: 现有点云补全工作都是1个input 1个output，忽略了未知区域的几何不确定性，即一个残缺点云，可能对应多个完整点云。


3. Unsupervised 3D Shape Completion through GAN Inversion（CVPR2021）[[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Unsupervised_3D_Shape_Completion_Through_GAN_Inversion_CVPR_2021_paper.pdf)[[code]](https://github.com/junzhezhang/shape-inversion)


Problem: 监督学习在in-domain数据表现好，但应用到其他数据时，由于存在domain gap，泛化能力差。


4. Cycle4Completion: Unpaired Point Cloud Completion using Cycle Transformation with Missing Region Coding  （CVPR2021）[[paper]](http://cgcad.thss.tsinghua.edu.cn/liuyushen/main/pdf/LiuYS_CVPR21_Cycle4Completion.pdf)[[code]](https://github.com/diviswen/Cycle4Completion)


Problem: 之前的工作只关注残缺到完整的支路，没有关注到完整到残缺的支路，导致效果不好。

#### 全监督
1. GRNet：Gridding Residual Network for Dense Point Cloud Completion （ECCV2020）[[paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540341.pdf)[[code]](https://github.com/hzxie/GRNet)


Problem：之前的工作都是直接处理点云，没有考虑结构和上下文信息，会造成细节上的损失。Voxelization会造成细节丢失，MLP的方法没有用好local的上下文信息

Contribution：提出一个3D Grid作为中间表示，将点云有序结构化，还提出Cubic Feature Sampling来提取上下文信息

2. Density-aware Chamfer Distance as a Comprehensive Metric for Point Cloud Completion（NIPS2021）[[paper]](https://arxiv.org/abs/2111.12702)[[code]](https://github.com/wutong16/Density_aware_Chamfer_Distance)

Problem：之前常用的CD，EMD loss有局限，CD对点集密度敏感，EMD对细节优化不好

Contribution：提出DCD loss，更高效的损失函数

3.  PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers （ICCV2021）[[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_PoinTr_Diverse_Point_Cloud_Completion_With_Geometry-Aware_Transformers_ICCV_2021_paper.pdf)[[code]](https://github.com/yuxumin/PoinTr)

Problem：之前基于MLP+max pooling的网络结构导致了很多细节损失，需要学习好的细节信息和long-range关系

Contribution：首个点云补全的Transformer，因为Transformer能良好处理长序列问题

4. Variational Relational Point Completion Network（CVPR2021）[[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Pan_Variational_Relational_Point_Completion_Network_CVPR_2021_paper.pdf)[[code]](https://github.com/paul007pl/VRCNet)

Problem：当前的方法倾向于生成全局的skeleton，但忽略了局部的细节
当前有利用local fature信息的方法是学习了一个 partila - complete的映射，缺乏基于局部观察的条件生成能力。之前的方法无法获得几何对称性、规则排列和表面平滑度等信息

5. PMP-Net: Point Cloud Completion by Learning Multi-step Point Moving Paths (CVPR 2021) [[paper]](http://cgcad.thss.tsinghua.edu.cn/liuyushen/main/pdf/LiuYS_CVPR21_PMP-Net.pdf)[[code]](https://github.com/diviswen/PMP-Net)

Problem：当前encoder-decoder结构的方法大多从一个latent code来生成点云，然而光是一个latent code的能力有限，不能得到很高质量的点云

Contribution：从新的视角看点云补全任务，通过移动现有点

6. SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer (ICCV 2021)  [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Xiang_SnowflakeNet_Point_Cloud_Completion_by_Snowflake_Point_Deconvolution_With_Skip-Transformer_ICCV_2021_paper.pdf)[[code]](https://github.com/AllenXiangX/SnowflakeNet)

Problem:由于点云离散和不规则的特性，之前的方法对局部细节处理不好

7. Point Cloud Completion by Skip-attention Network with Hierarchical Folding（CVPR2020）[[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wen_Point_Cloud_Completion_by_Skip-Attention_Network_With_Hierarchical_Folding_CVPR_2020_paper.pdf)

Problem:现有方法学到的Global feature 缺少local细节信息


8. Style-based Point Generator with Adversarial Rendering for Point Cloud Completion [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Xie_Style-Based_Point_Generator_With_Adversarial_Rendering_for_Point_Cloud_Completion_CVPR_2021_paper.pdf)[[code]](https://github.com/microsoft/SpareNet)

Problem: 之前的decoder常用的folding方法是目前的一个性能瓶颈，设计了一个基于feature的style base的Generatoor，CD，EMD监督信号不够，引入2D深度图loss

