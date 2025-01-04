# Classification: Training ResNet on CIFAR-10

This repository includes:

- Reference code
- Pre-trained models
- Analysis of the hyperparameter tuning process

The CIFAR-10 dataset can be downloaded using the following command:  
```bash
!wget "http://ai-atest.bj.bcebos.com/cifar-10-python.tar.gz" -O cifar-10-python.tar.gz
```

---

## Hyperparameter Tuning Process

This process involves adjusting the following hyperparameters to optimize model performance:

- Batch size  
- Optimizer  
- Regularization factor  
- Momentum coefficient (for the momentum optimizer)  
- Learning rate  
- Number of network layers  
- Number of epochs  
- Data augmentation techniques  

### Batch Size

The effect of different batch sizes (32, 64, 128, 512, and 1024) on training outcomes was analyzed to determine the optimal batch size.  

---

#### Experimental Setup:
- Optimizer: Momentum optimizer (momentum coefficient: 0.9)  
- Regularization factor: \(1 \times 10^{-4}\)  
- Learning rate: 0.1/0.01/0.001  
- Number of network layers: 20  
- Epochs: 10  
- Data augmentation: Enabled  

![Batch Size Results](./result/BatchSize.png)

The results indicate that evaluation accuracy improves with increasing batch size up to 512. Beyond this point, accuracy declines, suggesting that the optimal batch size for this experiment is 512.

---

### Optimizer and Regularization Factor

This experiment explores the performance of four optimizers—Momentum, Adam, SGD, and Adagrad—under different regularization factors. The goal is to identify the optimal optimizer-regularization factor combination.  

---

#### Experimental Setup:
- Batch size: 512  
- Momentum coefficient (if applicable): 0.9  
- Learning rate: 0.1/0.01/0.001  
- Number of network layers: 20  
- Epochs: 10  
- Data augmentation: Enabled  

![Optimizer Results](./result/Optimizer.png)

From the table, the optimal regularization factors are as follows:
- Momentum optimizer: \(1 \times 10^{-4}\)  
- Adam optimizer: No regularization  
- SGD optimizer: \(1 \times 10^{-5}\)  
- Adagrad optimizer: \(1 \times 10^{-4}\)  

The momentum optimizer with a regularization factor of \(1 \times 10^{-4}\) achieved the best performance and was selected for subsequent experiments.

---

### Momentum Coefficient

The experiment evaluated the effect of different momentum coefficients (0.9, 0.5, and 0.1) on model performance using the momentum optimizer.  

---

#### Experimental Setup:
- Batch size: 512  
- Optimizer: Momentum optimizer  
- Regularization factor: \(1 \times 10^{-4}\)  
- Learning rate: 0.1/0.01/0.001  
- Number of network layers: 20  
- Epochs: 10  
- Data augmentation: Enabled  

![Momentum Coefficient Results](./result/MomentumCoefficient.png)

The results show that a momentum coefficient of 0.9 yields the best experimental outcomes, which is therefore used in subsequent experiments.

---

### Learning Rate

This experiment investigates the impact of different learning rates on model performance. Four segmented learning rate schedules were tested:  
1. 0.05/0.005/0.0005  
2. 0.1/0.01/0.001  
3. 0.2/0.02/0.002  
4. 0.5/0.05/0.005  

---

#### Experimental Setup:
- Batch size: 512  
- Optimizer: Momentum optimizer  
- Momentum coefficient: 0.9  
- Regularization factor: \(1 \times 10^{-4}\)  
- Number of network layers: 20  
- Epochs: 10  
- Data augmentation: Enabled  

![Learning Rate Results](./result/LearningRate.png)

The optimal results were achieved with the 0.1/0.01/0.001 learning rate schedule, which was selected for subsequent experiments.

---

### Number of Network Layers

The influence of the number of layers on ResNet's performance was evaluated by testing ResNet-20, ResNet-32, ResNet-44, and ResNet-56.  

---

#### Experimental Setup:
- Batch size: 512  
- Optimizer: Momentum optimizer  
- Momentum coefficient: 0.9  
- Regularization factor: \(1 \times 10^{-4}\)  
- Learning rate: 0.1/0.01/0.001  
- Epochs: 10  
- Data augmentation: Enabled  

![Network Layers Results](./result/networkLayers.png)

ResNet-32 achieved the highest accuracy and was chosen as the network structure for subsequent experiments.

---

### Number of Epochs

The effect of different training epochs (10, 50, 100, and 150) was analyzed.  

---

#### Experimental Setup:
- Batch size: 512  
- Optimizer: Momentum optimizer  
- Momentum coefficient: 0.9  
- Regularization factor: \(1 \times 10^{-4}\)  
- Learning rate: 0.1/0.01/0.001  
- Number of network layers: 32  
- Data augmentation: Enabled  

![Epoch Results](./result/Epoch.png)

The results indicate that accuracy improves with the number of epochs but saturates after 100 epochs. Considering training costs, 150 epochs were selected for the final setup.

---

### Data Augmentation

The impact of data augmentation was assessed by comparing results from two experimental groups: one with data augmentation and one without.

---

#### Experimental Setup:
- Batch size: 512  
- Optimizer: Momentum optimizer  
- Momentum coefficient: 0.9  
- Regularization factor: \(1 \times 10^{-4}\)  
- Learning rate: 0.1/0.01/0.001  
- Number of network layers: 32  
- Epochs: 150  

![Data Augmentation Results](./result/DataAugmentation.png)

Data augmentation improved accuracy by approximately 10%, reducing overfitting caused by the small training dataset size.

- **With data augmentation:**  
  ![With Augmentation](./result/1.png)

- **Without data augmentation:**  
  ![Without Augmentation](./result/2.png)