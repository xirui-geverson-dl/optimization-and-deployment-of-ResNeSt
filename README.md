# optimization-and-deployment-of-ResNeSt
**Optimization and Deployment Analysis of Split-Attention Network (ResNeSt)**

Final project for NYU CSCI-GA 3033-091 Introduction to Deep Learning Systems

Group members: Haodong Ge, Xirui Fu
## Abstract
In this project, 50-LayerSplit-Attention Network (ResNeSt50) is initially implemented with original bottleneck structure on classificationtask using CIFAR10 dataset. To attempt optimizing the model, we innovatively implement split-attention technique on basic block structure. For comparison between two structures,both ResNeSt50  models are trained at different GPU types.Then, a transfer learning of both models to the CIFAR100 dataset is done to further verify comparison results. For the deployment analysis,  the winner model from the previous part is  converted  and  deployedin both Pytorch and Caffe2 frameworks. After model inferences, two frameworks are compared from aspects such as test accuracy, infer-ence time and model sizes.
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

## Models

### ResNeSt training 
```resnest_training.ipynb```
- Split-attention architecture 
- Basic block and Bottlebeck: building block framework 
- Training function & configuration

### Transfer learning 
```transfer_learning.ipynb:```
- Pretrained models of basic block ResNeSt-50 :\
 https://drive.google.com/file/d/1-0EOTYj_IgORuzDT2ZAh3H1adbbIAmJ3/view?usp=sharing
- Load pretrained ResNeSt-50 and transfer to CIFAR100 dataset with finetunning method. 

### Results
* Accuracy

|          | K80 | P100 | V100 | Average | Transfer learning |
|----------|-------------------|-------------------|-------------------|-------------------|-----------------|
| Basic Block  | 0.8862               | 0.8923           | 0.8907            | 0.8897           | 0.6323             |
| Bottleneck   | 0.8705              | 0.8742           | 0.8765            | 0.8737           | 6152             |

* Training time (s/epoch)

|          | K80 | P100 | V100 |
|----------|-------------------|-------------------|-------------------|
| Basic Block  | 302               | 36           | 25.5            | 
| Bottleneck   | 149             | 31           | 22           | 

* Test accuracy during training

![image](https://user-images.githubusercontent.com/71922238/118348223-7399db80-b57b-11eb-9bee-9a0307377a24.png)



## Deployment Analysis 

### **Convert_torch_caffe2.ipynb**: 

- Loading pretrained pytorch model and transfer to ONNX\
Tranfered ONNX model: https://drive.google.com/file/d/1-Bf3XDk4crSTipXZdtGkCPQ8rW1KxMdw/view?usp=sharing
- Transfer ONNX model to Caffe2
- Simulate Caffe2 deplyment with ONNX model on Caffe2 Backend 
- Inference on single image and testsets 

### Deployment Results
- Inference of Caffe2 model on single image 
<img width="661" alt="Screen Shot 2021-05-15 at 10 28 40 AM" src="https://user-images.githubusercontent.com/61107669/118345629-5f001800-b568-11eb-99a1-0b032d84b8aa.png">

- Evaluation of traditional and mobile DL framework on test sets 

|          | Test_acc small(%) | Avg_time small(s) | Test_acc large(%) | Avg_time Large(s) | Model_size (mb) |
|----------|-------------------|-------------------|-------------------|-------------------|-----------------|
| Pytorch  | 0.9               | 0.05145           | 0.8923            | 0.05242           | 128             |
| Caffe2   | 0.88              | 0.02854           | 0.8812            | 0.02878           | 115             |

## Reference
ResNeSt originally adapted from: https://github.com/zhanghang1989/ResNeSt

Transfer Learning tutorial from : https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

Pytorch-Caffe2 conversion blog: https://learnopencv.com/pytorch-model-inference-using-onnx-and-caffe2/

Other Githubs used for reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
2. https://github.com/facebookarchive/tutorials/blob/master/Multi-GPU_Training.ipynb
