# optimization-and-deployment-of-ResNeSt
Final project for 3033 Intro to deep learning systems
## Abstract
In this project, 50-LayerSplit-Attention Network (ResNeSt50) is initially implemented with original bottleneck structure on classificationtask using CIFAR10 dataset. To attempt optimizing the model, we innovatively implementsplit-attention technique on basic block struc-ture. For comparison between two structures,both ResNeSt50  models are trained at differ-ent GPU types.Then, a transfer learning of both models to the CIFAR100 dataset is doneto further verify comparison results. For thedeployment analysis,  the winner model fromthe previous part is  converted  and  deployedin both Pytorch and Caffe2 frameworks. After model inferences, two frameworks are compared from aspects such as test accuracy, infer-ence time and model sizes.
## Diagram 
<img width="793" alt="Screen Shot 2021-05-15 at 9 39 59 AM" src="https://user-images.githubusercontent.com/61107669/118344598-90291a00-b561-11eb-954e-5cb6ab179420.png">
## Getting Started
### Library
- [Pytorch](https://pytorch.org)
* [ONNX](https://onnx.ai)
* [Laravel](https://caffe2.ai)
