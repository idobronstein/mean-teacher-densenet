#!/usr/bin/env bash

echo "Downloading CIFAR-100"
mkdir -p data/images/cifar/cifar100
(
    cd data/images/cifar/cifar100
    curl -O 'https://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz'
    tar xvzf cifar-100-matlab.tar.gz
    mv cifar-100-matlab/* .
    rmdir cifar-100-matlab
)

echo
echo "Preprocessing CIFAR-100"
python datasets/preprocess_cifar100.py

echo
echo "All done!"
