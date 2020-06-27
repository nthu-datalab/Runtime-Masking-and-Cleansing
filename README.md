# Runtime-Masking-and-Cleansing (RMC)
This is the repo for "Adversarial Robustness via Runtime Masking and Cleansing", Yi-Hsuan Wu, Chia-Hung Yuan, and Shan-Hung Wu, In Proceedings of ICML 2020. Our code is implemented in TensorFlow 2.0 using all the best practices.

We devise a new defense method, called runtime masking and cleansing (RMC), to improve adversarial robustness. RMC adapts the network at runtime before making a prediction to dynamically mask network gradients and cleanse the model of the non-robust features inevitably learned during the training process due to the size limit of the training set.

The following figure illustrates the defense mechanism in RMC:

<p align="center">
	<img src="./figures/rmc-architecture-flow-new.png" width=600>
</p>

1. Augment dataset with adversarial examples
2. Find K-nearest neighbors (KNN) of test data from the augmented dataset
3. Adapt the network with KNN
4. Make predictions

## Installation
Clone and install requirements.
```
git clone https://github.com/nthu-datalab/Runtime-Masking-and-Cleansing.git
cd Runtime-Masking-and-Cleansing
pip install -r requirements.txt
```

## Usage
RMC works well with any existing model architecture. We use pretrained ResNet-152v2 in our code as an example. Before running ```main.py```, we have to run the following command to create augmented datasets
```
cd prepare
python augment_dataset.py
```