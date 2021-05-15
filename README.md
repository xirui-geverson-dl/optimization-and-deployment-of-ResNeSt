# optimization-and-deployment-of-ResNeSt
Optimization and Deployment Analysis of Split-Attention Network (ResNeSt)

Final project for NYU CSCI-GA 3033-091 Introduction to Deep Learning Systems
## Abstract
In this project, 50-LayerSplit-Attention Network (ResNeSt50) is initially implemented with original bottleneck structure on classificationtask using CIFAR10 dataset. To attempt optimizing the model, we innovatively implementsplit-attention technique on basic block struc-ture. For comparison between two structures,both ResNeSt50  models are trained at differ-ent GPU types.Then, a transfer learning of both models to the CIFAR100 dataset is doneto further verify comparison results. For thedeployment analysis,  the winner model fromthe previous part is  converted  and  deployedin both Pytorch and Caffe2 frameworks. After model inferences, two frameworks are compared from aspects such as test accuracy, infer-ence time and model sizes.
## Diagram 
<img width="793" alt="Screen Shot 2021-05-15 at 9 39 59 AM" src="https://user-images.githubusercontent.com/61107669/118344598-90291a00-b561-11eb-954e-5cb6ab179420.png">

## Dependencies

### Library
* Python 3.6+
* Pandas
* OpenCV
* [Pytorch](https://pytorch.org)
* [ONNX](https://onnx.ai)
* [Caffe2](https://caffe2.ai)

### Install
Note that Caffe2 package need to be installed based on conda environment.
```bash
! pip install onnx==1.8.1
! conda install pytorch-nightly-cpu -c pytorch
```

## Load Datasets
There are two files in dataload file. CIFAR10 is used in model training for both ResNeSt building frameworks and CIFAR100 is used on transfer learning on pretraiend ResNeSt-50 models. 
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)

These two classic image classification datasets can be loaded easily from torchvision 
```bash
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
```

## Models

### ResNeSt training 
resnest_training.ipynb: 
- Split-attention architecture 
- Basic block and Bottlebeck: building block framework 
- Training function & configuration

### Transfer learning 
