# Continual/Lifelong/Incremental Learning

Please feel free to pull requests or open an issue to add papers. 

## Quick links:
- [Surevey](#Surevey)
- [Replay Methods](#Replay-Methods)
- [Regularization-based Methods](#Regularization-Methods)
- [Dynamic architecture methods](#Dynamic-architecture-Methods)
- [Others](#Others)

## Surevey 
- [Class-incremental learning: survey and performance evaluation on image classification. (2021)](https://arxiv.org/abs/2010.15277) 
- [Continual learning: A comparative study on how to defy forgetting in classification tasks. (2020)](https://arxiv.org/pdf/1909.08383.pdf) **(Recommend read)**
- [Continual Lifelong Learning with Neural Networks: A Review. (2019)](https://arxiv.org/pdf/1802.07569.pdf)
- [Continual Lifelong Learning in Natural Language Processing: A Survey. (2020)](https://www.aclweb.org/anthology/2020.coling-main.574.pdf)
- [Embracing Change: Continual Learning in Deep Neural Networks. (2020)](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(20)30219-9) 

## Replay Methods
#### 2021
| Title    | Venue       | Code     |
|:-------|:--------:|:-------:|
| [Self-Supervised Training Enhances Online Continual Learning](https://arxiv.org/pdf/2103.14010.pdf) | - | -| 
| [Rehearsal revealed: The limits and merits of revisiting samples in continual learning](https://arxiv.org/pdf/2104.07446.pdf)| -| [Pyotch(Author)](https://github.com/Mattdl/RehearsalRevealed)| 

#### 2020
| Title    | Venue       | Code     |
|:-------|:--------:|:-------:|
| [Online Continual Learning from Imbalanced Data](https://proceedings.icml.cc/static/paper_files/icml/2020/4727-Paper.pdf) | ICML | - |
| [Orthogonal Gradient Descent for Continual Learning](http://arxiv.org/abs/1910.07104)  | AISTATS    | - |
| [Meta-Consolidation for Continual Learning](https://papers.nips.cc/paper/2020/file/a5585a4d4b12277fee5cad0880611bc6-Paper.pdf) | NeurIPS | [Pyotch(Author)](https://github.com/JosephKJ/merlin) |
| [Coresets via Bilevel Optimization for Continual Learning and Streaming](http://arxiv.org/abs/2006.03875) | NeurIPS | [Pyotch(Author)](https://github.com/zalanborsos/bilevel_coresets/tree/b628fdde1db83151bf560b31c8c4d23279552678) |
| [Dark Experience for General Continual Learning: a Strong, Simple Baseline](https://papers.nips.cc/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf) | NeurIPS | [Pyotch(Author)](https://github.com/aimagelab/mammoth) |
| [La-MAML: Look-ahead Meta Learning for Continual Learning](https://arxiv.org/pdf/2007.13904.pdf) | NeurIPS | [Pyotch(Author)](https://github.com/montrealrobotics/La-MAML) |
| [Remembering for the Right Reasons: Explanations Reduce Catastrophic Forgetting](https://arxiv.org/pdf/2010.01528.pdf) | - | [Author](https://github.com/SaynaEbrahimi/Remembering-for-the-Right-Reasons)-check later |
| [iTAML : An Incremental Task-Agnostic Meta-learning Approach](https://arxiv.org/pdf/2003.11652.pdf) | CVPR | [Pyotch(Author)](https://github.com/brjathu/iTAML) | 
| [Bilevel Continual Learning](https://arxiv.org/pdf/2007.15553.pdf) | - | [Pyotch(Author)](https://github.com/phquang/Bilevel-Continual-Learning) | 
| [Continual Deep Learning by Functional Regularisation of Memorable Past](https://arxiv.org/pdf/2004.14070.pdf) |  NeurIPS | [Pyotch(Author)](https://github.com/team-approx-bayes/fromp) |



#### 2019
| Title    | Venue       | Code     |
|:-------|:--------:|:-------:|
| [Episodic Memory in Lifelong Language Learning](https://papers.nips.cc/paper/9471-episodic-memory-in-lifelong-language-learning.pdf) | NeurIPS | - | 
| [Experience Replay for Continual Learning](https://arxiv.org/pdf/1811.11682.pdf) | NeurIPS | - | 
| [Online Continual Learning with Maximally Interfered Retrieval](https://papers.nips.cc/paper/2019/file/15825aee15eb335cc13f9b559f166ee8-Paper.pdf) | NeurIPS | [Pyotch(Author)](https://github.com/optimass/Maximally_Interfered_Retrieval)  |
| [Gradient based sample selection for online continual learning](https://papers.nips.cc/paper/2019/file/e562cd9c0768d5464b64cf61da7fc6bb-Paper.pdf) |  NeurIPS | [Pyotch(Author)](https://github.com/rahafaljundi/Gradient-based-Sample-Selection) |
| [Meta-Learning Representations for Continual Learning](https://papers.nips.cc/paper/2019/file/f4dd765c12f2ef67f98f3558c282a9cd-Paper.pdf) | NeurIPS | [Pyotch(Author)](https://github.com/khurramjaved96/mrcl) |
| [Learning to learn without forgetting by maximizing transfer and minimizing interference](https://arxiv.org/pdf/1810.11910.pdf) | ICLR | [Pyotch(Author)](https://github.com/mattriemer/mer)  | 
| [IL2M: Class incremental learning with dual memory](https://openaccess.thecvf.com/content_ICCV_2019/papers/Belouadah_IL2M_Class_Incremental_Learning_With_Dual_Memory_ICCV_2019_paper.pdf) | ICCV |  [Pyotch(Author)](https://github.com/EdenBelouadah/class-incremental-learning/tree/master/il2m) |
| [Overcoming catastrophic forgetting for continual learning via model adaptation](https://openreview.net/pdf?id=ryGvcoA5YX) | ICLR | [Tensorflow(Author)](https://github.com/morning-dews/PGMA_tensorflow) |
| [Learning a unified classifier incrementally via rebalancing](http://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.pdf) | CVPR | [Pytorch](https://github.com/hshustc/CVPR19_Incremental_Learning) |
| [Task Agnostic Continual Learning via Meta Learning](https://openreview.net/attachment?id=AeIzVxdJgeb&name=pdf)| ICML-w | - |
| [On Tiny Episodic Memories in Continual Learning](https://arxiv.org/pdf/1902.10486.pdf) | ICML-w | [Tensorflow(Author)](https://github.com/facebookresearch/agem) |
| [Efficient Lifelong Learning with A-GEM](https://arxiv.org/pdf/1812.00420.pdf) | ICLR | [Tensorflow(Author)](https://github.com/facebookresearch/agem) |
| [Incremental Learning Using Conditional Adversarial Networks](https://openaccess.thecvf.com/content_ICCV_2019/papers/Xiang_Incremental_Learning_Using_Conditional_Adversarial_Networks_ICCV_2019_paper.pdf) | ICCV| - |
#### 2018
| Title    | Venue       | Code     |
|:-------|:--------:|:-------:|


#### 2017 and Before
| Title    | Venue       | Code     |
|:-------|:--------:|:-------:|
| [Gradient Episodic Memory for Continual Learning](https://arxiv.org/abs/1706.08840) | NeurIPS | [Pytorch(Author)](https://github.com/facebookresearch/GradientEpisodicMemory) |
| [Continual Learning with Deep Generative Replay](https://arxiv.org/pdf/1705.08690.pdf) | NeurIPS | [Pytorch](https://github.com/kuc2477/pytorch-deep-generative-replay)  |
| [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/pdf/1611.07725.pdf) | CVPR | [Theano(Author)](https://github.com/srebuffi/iCaRL)  |


## Regularization Methods
#### 2021
| Title    | Venue       | Code     |
|:-------|:--------:|:-------:|
| [Linear Mode Connectivity in Multitask and Continual Learning](https://openreview.net/pdf?id=Fmg_fQYUejf) | ICLR(Under review) | [Author-check later](https://github.com/imirzadeh/MC-SGD)  | 
| [Gradient Projection Memory for Continual Learning](https://openreview.net/pdf?id=3AOj0RCNC2) | ICLR | [Pytorch(Author)](https://github.com/sahagobinda/GPM)| 
| [Unifying Regularisation Methods for Continual Learning](https://arxiv.org/pdf/2006.06357v2.pdf) | - |[Tensorflow](https://github.com/freedbee/continual_regularisation)| 

#### 2020
| Title    | Venue       | Code     |
|:-------|:--------:|:-------:|
| [Sliced Cramércram´Cramér Synaptic Consolidation for Preserving Deeply Learned Representations](https://openreview.net/pdf?id=BJge3TNKwH) | ICLR | - |
| [Continual Learning with Node-Importance based Adaptive Group Sparse Regularization](https://papers.nips.cc/paper/2020/file/258be18e31c8188555c2ff05b4d542c3-Paper.pdf) | NeurIPS | - |
| [Continual Learning with Adaptive Weights (CLAW)](https://openreview.net/pdf?id=Hklso24Kwr) | ICLR | - |

#### 2019
| Title    | Venue       | Code     |
|:-------|:--------:|:-------:|
| [Uncertainty-based continual learning with adaptive regularization](http://arxiv.org/abs/1905.11614) | NeurIPS | [Pytorch(Author)](https://github.com/csm9493/UCL) |
| [Task-Free Continual Learning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Aljundi_Task-Free_Continual_Learning_CVPR_2019_paper.pdf) | CVPR | - |
| [Continual Learning by Asymmetric Loss Approximation with Single-Side Overestimation](https://arxiv.org/pdf/1908.02984.pdf) | ICCV | [Tensorflow](https://github.com/dmpark04/alasso) |
| [Learning without Memorizing](https://openaccess.thecvf.com/content_CVPR_2019/papers/Dhar_Learning_Without_Memorizing_CVPR_2019_paper.pdf) | CVPR | [Pytorch](https://github.com/stony-hub/learning_without_memorizing) |
| [Selfless Sequential Learning](https://arxiv.org/pdf/1806.05421.pdf) | ICLR |  [Pytorch(Author)](https://github.com/rahafaljundi/Selfless-Sequential-Learning)|


#### 2018
| Title    | Venue       | Code     |
|:-------|:--------:|:-------:|
| [Memory Aware Synapses: Learning What (not) to Forget](https://arxiv.org/pdf/1711.09601.pdf) | ECCV | [Pytorch](https://github.com/wannabeOG/MAS-PyTorch) |
| [Progress & Compress: A scalable framework for continual learning](https://arxiv.org/pdf/1805.06370.pdf) | ICML | - |
| [Online structured laplace approximations for overcoming catastrophic forgetting](https://papers.nips.cc/paper/7631-online-structured-laplace-approximations-for-overcoming-catastrophic-forgetting.pdf) |  NeurIPS | [Pytorch(Author)](https://github.com/hannakb/KFA) | 


#### 2017 and Before
| Title    | Venue       | Code     |
|:-------|:--------:|:-------:|
| [Learning without Forgetting](https://arxiv.org/pdf/1606.09282.pdf)    | ECCV | [Matlab(Author)](https://github.com/lizhitwo/LearningWithoutForgetting)  |
| [Overcoming catastrophic forgetting in neural network](https://arxiv.org/pdf/1612.00796.pdf) | PNAS | [TensorFlow](https://github.com/ariseff/overcoming-catastrophic) |
| [Continual learning through synaptic intelligence](https://arxiv.org/pdf/1703.04200.pdf) | ICML | [TensorFlow](https://github.com/ganguli-lab/pathint) |


## Dynamic architecture Methods
#### 2021 
| Title    | Venue       | Code     |
|:-------|:--------:|:-------:|
| [Efficient Continual Learning with Modular Networks and Task-Driven Priors](https://arxiv.org/pdf/2012.12631.pdf) | ICLR | [Pytorch(Author)](https://github.com/TomVeniat/MNTDP)|
| [Efficient Feature Transformations for Discriminative and Generative Continual Learning](https://arxiv.org/pdf/2103.13558.pdf) | CVPR |[check later(Author)](https://github.com/vkverma01/EFT)| 

#### 2020 
| Title    | Venue       | Code     |
|:-------|:--------:|:-------:|
| [Conditional Channel Gated Networks for Task-Aware Continual Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Abati_Conditional_Channel_Gated_Networks_for_Task-Aware_Continual_Learning_CVPR_2020_paper.pdf) | CVPR | - |


#### 2019 
| Title    | Venue       | Code     |
|:-------|:--------:|:-------:|
| [Learning to remember: A synaptic plasticity driven framework for continual learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ostapenko_Learning_to_Remember_A_Synaptic_Plasticity_Driven_Framework_for_Continual_CVPR_2019_paper.pdf) | CVPR | [Pytorch](https://github.com/SAP-samples/machine-learning-dgm) | 
| [Large Scale Incremental Learning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.pdf) | CVPR | [Tensorflow(Author)](https://github.com/wuyuebupt/LargeScaleIncrementalLearning) |
| [Compacting, Picking and Growing for Unforgetting Continual Learning](https://arxiv.org/pdf/1910.06562.pdf) | NeurIPS | [Pytorch(Author)](https://github.com/ivclab/CPG) | 
| [Random Path Selection for Incremental Learning](https://papers.nips.cc/paper/2019/file/83da7c539e1ab4e759623c38d8737e9e-Paper.pdf) | NeurIPS | [Pytorch(Author)](https://github.com/brjathu/RPSnet) |


#### 2018
| Title    | Venue       | Code     |
|:-------|:--------:|:-------:|
| [PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning](https://arxiv.org/pdf/1711.05769.pdf) | CVPR | [Pytorch(Author)](https://github.com/arunmallya/packnet) |
| [Overcoming Catastrophic Forgetting with Hard Attention to the Task](https://arxiv.org/pdf/1801.01423.pdf) |  ICML |   [Pytorch(Author)](https://github.com/joansj/hat)  |
| [Lifelong Learning with Dynamically Expandable Networks](https://openreview.net/pdf?id=Sk7KsfW0-) | ICLR | [Tensorflow(Author)](https://github.com/jaehong-yoon93/DEN) |
|[Alleviating catastrophic forgetting using context-dependent gating and synaptic stabilization](https://www.pnas.org/content/115/44/E10467)  | PNAS | [Tensorflow(Author)](https://github.com/nmasse/Context-Dependent-Gating) |
| [FearNet: Brain-Inspired Model for Incremental Learning](https://arxiv.org/pdf/1711.10563.pdf) | ICLR | - |
#### 2017 and Before
| Title    | Venue       | Code     |
|:-------|:--------:|:-------:|
| [Encoder Based Lifelong Learning](https://openaccess.thecvf.com/content_ICCV_2017/papers/Rannen_Encoder_Based_Lifelong_ICCV_2017_paper.pdf) | ICCV | [Pytorch(Author)](https://github.com/rahafaljundi/Pytorch-implementation-of-Encoder-Based-Lifelong-learning) |

## Others
#### 2021
| Title    | Venue       | Code     |
|:-------|:--------:|:-------:|
| [CLeaR: An Adaptive Continual Learning Framework for Regression Tasks](https://arxiv.org/pdf/2101.00926.pdf)| -| -|
| [A Theoretical Analysis of Catastrophic Forgetting through the NTK Overlap Matrix](https://arxiv.org/pdf/2010.04003.pdf) | AISTATS | - |

#### 2020
| Title    | Venue       | Code     |
|:-------|:--------:|:-------:|
|[Understanding the Role of Training Regimes in Continual Learning](https://papers.nips.cc/paper/2020/file/518a38cc9a0173d0b2dc088166981cf8-Paper.pdf) | NeurIPS | [Pytorch(Author)](https://github.com/imirzadeh/stable-continual-learning) |
| [Optimal Continual Learning has Perfect Memory and is NP-hard](https://arxiv.org/pdf/2006.05188.pdf) | ICML | - | 
| [Continual Learning from the Perspective of Compression](https://arxiv.org/pdf/2006.15078.pdf) | ICML-w |  - | 
| [Online Fast Adaptation and Knowledge Accumulation: a New Approach to Continual Learning](https://arxiv.org/pdf/2003.05856.pdf) | NIPS | [Pytorch(Author)](https://github.com/ElementAI/osaka) |

#### 2019
| Title    | Venue       | Code     |
|:-------|:--------:|:-------:|
| [Reconciling meta-learning and continual learning with online mixtures of tasks](https://papers.nips.cc/paper/9112-reconciling-meta-learning-and-continual-learning-with-online-mixtures-of-tasks.pdf) | NeurIPS | - | 
| [A comprehensive, application-oriented study of catastrophic forgetting in DNNs](https://arxiv.org/pdf/1905.08101.pdf) | ICLR| [Tensorflow(Author)](https://gitlab.informatik.hs-fulda.de/ML-Projects/CF_in_DNNs) |
| [Incremental Object Learning from Contiguous Views](https://openaccess.thecvf.com/content_CVPR_2019/papers/Stojanov_Incremental_Object_Learning_From_Contiguous_Views_CVPR_2019_paper.pdf) | CVPR | [Author](https://iolfcv.github.io)|
| [Backpropamine: training self-modifying neural networks with differentiable neuromodulated plasticity](https://arxiv.org/pdf/2002.10585.pdf) | ICLR | - | 




#### 2017 and Before
| Title    | Venue       | Code     |
|:-------|:--------:|:-------:|
| [An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks](https://arxiv.org/pdf/1312.6211.pdf) | - | [Theano(Author)](https://github.com/goodfeli/forgetting) |
# License

----
MIT

