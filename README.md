# MU-MAE: Multimodal Masked Autoencoders-Based One Shot learning

Challenges arise in accurately recognizing human activities using multimodal sensors. These challenges stem from labor-intensive data collection and annotation, as well as dependence on external pretrained models or additional data, contributing to overall inefficiency. In response, we present Multimodal Masked Autoencoders-Based One Shot Learning (Mu-MAE), an effective and efficient multimodal one-shot classification model guided by multimodal masked autoencoders. Mu-MAE addresses these challenges by employing a one-shot multimodal learning strategy, significantly reducing annotation costs, and incorporating a multimodal masked autoencoder approach with a synchronized masking strategy specifically for wearable sensors, enabling efficient self-supervised pretraining without external dependencies. Furthermore, Mu-MAE leverages the representation extracted from multimodal masked autoencoders as prior information input to a cross-attention multimodal fusion layer. This fusion layer emphasizes spatiotemporal features requiring attention across different modalities while highlighting differences from other classes, aiding in the classification of various classes in metric-based one-shot learning. Comprehensive evaluations on MMAct one-shot classification show that Mu-MAE outperforms all the evaluated approaches, achieving up to an 80.17% accuracy for five-way one-shot multimodal classification, without the use of additional data.

# Getting started

**Environment**:
1. Anaconda with python >= 3.8
2. Pytorch >= 1.8.1
3. Torchvision >= 0.9.1
4. Tensorflow >=2.2.0
5. timm = 0.4.8/0.4.12
6. deepspeed = 0.5.8
7. TensorboardX
8. decord
9. einops



**Few-shot splits**:

We first introduce one-shot data split for [MMAct](https://mmact19.github.io/2019/),  please refer to  ./one-shot-finetuning/save/video_datasets/splits/.

# Train and eval

Mu-MAE involves two distinct steps. To understand the pretraining process, consult the README.md file located at ./mu-mae-pretrain/. For insights into the one-shot finetuning process, refer to the README.md file at ./one-shot-finetuning/.

# References
This algorithm library is extended from [VideoMAE](https://github.com/MCG-NJU/VideoMAE), [MASTAF](https://anonymous.4open.science/r/STAF-30CF1/README.md), [TRX](https://github.com/tobyperrett/trx) and [Cross-attention](https://github.com/blue-blue272/fewshot-CAN), which builds upon several existing publicly available code:  [CNAPs](https://github.com/cambridge-mlg/cnaps), [torch_videovision](https://github.com/hassony2/torch_videovision) and [R3D Backbone](https://github.com/kenshohara/3D-ResNets-PyTorch)
