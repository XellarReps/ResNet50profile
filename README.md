# ResNet50 with profile

## Description
This repository contains code for training the ResNet50 model as well as profiling the graph calculating the model through forward hooks and MLPerf logging.

### Instructions for running the code
In the root of the project, run:
```bash
docker build -t resnet50 .
```
You can replace the base image in the Dockerfile with the desired supported one (for example, rocm/pytorch)

This project uses a dataset ILSVRC ImageNet

    O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma,
    Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, et al. Imagenet
    large scale visual recognition challenge. arXiv:1409.0575, 2014.

You need to download the dataset from the link (<DATASET_DIR>): https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data

In the root of the project, create a directory for the MLPerf log:
```bash
mkdir <RES_DIR_NAME>
```

Next, to launch the container:
```bash
docker run --ipc=host -it --rm --runtime=nvidia -v <DATASET_DIR>:/data/train -v <RES_DIR_NAME>:/results resnet50:latest
```

To start training, enter:
```bash
bash run_profile.sh
```

## Links
MLCommons (MLPerf logging):\
https://github.com/mlcommons/logging

The basic implementation of ResNet50 is taken from this link:\
https://moiseevigor.github.io/software/2022/12/18/one-pager-training-resnet-on-imagenet/
