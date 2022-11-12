# Classification:Training ResNet on CIFAR10
This repository contains:

* Dataset CIFAR10
* Reference code
* Trained models
* Analysis of the tuning process

## Tuning process
In the process, I will adjust the following hyperparameters：

* Batch size
* Optimizer
* Regularization factor
* Momentum coefficient when using Momentum Optimizer
* Learning rate
* Number of network layers
* Epoch
* Data Augmentation

### Batch size
In this section, we will try out batch sizes of 32, 64, 128, 512 and 1024, and analyse the effect of different batch sizes on the training effect and choose the optimal batch size. 

* * *

The other hyperparameter arrangements for the following experiments are
Optimizer: momentum optimizer, momentum coefficient: 0.9
Regularization factor: 1e-4
Learning rate: 0.1/0.01/0.001
Number of network layers: 20
Epoch: 10
Data enhancement: yes

![image](https://github.com/Lipyu/ResNetOnCIFAR10/blob/main/Result/BatchSize.png)

In the table we can see that the best evaluation accuracy increases as the batch size increases, however, after 512 there is a drop in accuracy, so we can assume that the best batch size in this experiment is 512。

### Optimizer and Regularization factor
As different optimisers use different optimal regularisation strategies, the performance of each of the four optimisers Momentum, Adam, SGD and Adagrad with different regularisation factors will be explored in this experiment, and the optimal results will be selected for comparison to choose the most suitable optimiser with the matching regularisation factor.

* * *

The other hyperparameter arrangements for the following experiments are
Batch size:512
Momentum coefficient: 0.9(if Momentun is used)
Learning rate: 0.1/0.01/0.001
Number of network layers: 20
Epoch: 10
Data enhancement: yes

![image](https://github.com/Lipyu/ResNetOnCIFAR10/blob/main/Result/Optimizer.png)

From the results in the table it can be seen that the momentum, Adam, SGD and Adagrad achieve optimality with regularisation factors of, 1e-4,no optimisation strategy,1e-5,1e-4 respectively. The best results were obtained with a regularization factor of 1e-4 for the momentum optimizer, so the momentum optimizer was chosen for the next experiments with a regularization factor of 1e-4.

### Momentum coefficient
Next the effect of choosing different momentum coefficients(0.9, 0.5, 0.1) on the experimental results when using the momentum optimiser will be experimented with.

* * *

The other hyperparameter arrangements for the following experiments are 
Batch size:512
Optimizer: momentum optimizer
Regularization factor: 1e-4
Learning rate: 0.1/0.01/0.001
Number of network layers: 20
Epoch: 10
Data enhancement: yes

![image](https://github.com/Lipyu/ResNetOnCIFAR10/blob/main/Result/MomentumCoefficient.png)

From the experimental results in the table, we can see that the best experimental results can be obtained when the momentum coefficient is 0.9, so we choose 0.9 as the momentum coefficient for the subsequent experiments。

### Learning rate
In this section, I will experiment with the effect of different learning rates on the results of the experiment. Since we are using segmented learning rates, four sets of learning rates are chosen: 0.05/0.005/0.0005,0.1/0.01/0.001, 0.2/0.02/0.002, 0.5/0.05/0.005

* * *

The other hyperparameter arrangements for the following experiments are 
Batch size:512
Optimizer: momentum optimizer
Momentum coefficient: 0.9
Regularization factor: 1e-4
Number of network layers: 20
Epoch: 10
Data enhancement: yes

![image](https://github.com/Lipyu/ResNetOnCIFAR10/blob/main/Result/LearningRate.png)

As can be seen from the data in the table, the experiment achieved optimal results at a learning rate of 0.1/0.01/0.001, so the learning rate of 0.1/0.01/0.001 group was chosen as the learning rate for the subsequent experiment.

### Number of network layers
In this experiment, we will explore the effect of different network layers of ResNet on the experimental results. Here we will experiment with ResNet20, ResNet32, ResNet44, ResNet56.

* * *

The other hyperparameter arrangements for the following experiments are ：
Batch size:512
Optimizer: momentum optimizer
Momentum coefficient: 0.9
Regularization factor: 1e-4
Learning rate: 0.1/0.01/0.001
Epoch: 10
Data enhancement: yes

![image](https://github.com/Lipyu/ResNetOnCIFAR10/blob/main/Result/networkLayers.png)

From the experimental results in the table above, it can be seen that the highest accuracy rate in this experiment was achieved when using ResNet32, and therefore ResNet32 will be used as our network structure in subsequent experiments.

### Epoch
Here we will experiment with the effect of different Epochs on the experimental results. The different Epochs used are 10, 50, 100, 150.

* * *

The other hyperparameter arrangements for the following experiments are ：
Batch size:512
Optimizer: momentum optimizer
Momentum coefficient: 0.9
Regularization factor: 1e-4
Learning rate: 0.1/0.01/0.001
Number of network layers: 32
Data enhancement: yes

![image](https://github.com/Lipyu/ResNetOnCIFAR10/blob/main/Result/Epoch.png)

As can be seen from the results in the table, the accuracy of the experimental results increases as the Epoch continues to increase, but also tends to saturate after exceeding 100. Considering the training cost, 150 will be used as the training Epoch in subsequent experiments.

### Data Augmentation
In this experiment, we will investigate the effect of data augmentation on the results of the experiment, so two sets of experiments are set up to investigate using data augmentation and not using data augmentation.

* * *

Batch size:512
Optimizer: momentum optimizer
Momentum coefficient: 0.9
Regularization factor: 1e-4
Learning rate: 0.1/0.01/0.001
Number of network layers: 32
Epoch: 150

(Group No.1 used data enhancement while Group No.2 did not)
![image](https://github.com/Lipyu/ResNetOnCIFAR10/blob/main/Result/DataAugmentation.png)

As you can see from the experimental data in the table, the difference in accuracy between using data augmentation and not using data augmentation is as much as 10 percent. The group without data augmentation showed more severe overfitting during the experiment, which was caused by the small size of our training data set. Therefore, we can conclude that data augmentation can reduce the overfitting phenomenon.

* With data augmentation:
![image](https://github.com/Lipyu/ResNetOnCIFAR10/blob/main/Result/1.png)


* Without data augmentation:
![image](https://github.com/Lipyu/ResNetOnCIFAR10/blob/main/Result/2.png)

