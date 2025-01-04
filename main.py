import warnings
warnings.filterwarnings('ignore')

import paddle
import numpy as np
import matplotlib.pyplot as plt
from paddle.static import InputSpec
from paddle.regularizer import L2Decay
from paddle import nn
from paddle.vision.models.resnet import BasicBlock
import paddle.vision.transforms as T
from paddle.io import DataLoader

# Automatic device detection with fallback to CPU
if paddle.is_compiled_with_cuda():
    try:
        paddle.set_device('gpu')
        print("Using GPU for training.")
    except:
        paddle.set_device('cpu')
        print("GPU is unavailable or incompatible, falling back to CPU.")
else:
    paddle.set_device('cpu')
    print("Using CPU for training.")

# Definition of the ResNet model
class ResNet(nn.Layer):
    def __init__(self, block, depth, num_classes=10):
        super(ResNet, self).__init__()
        layer_cfg = {
            20: [3, 3, 3],
            32: [5, 5, 5],
            44: [7, 7, 7],
            56: [9, 9, 9],
            110: [18, 18, 18],
            1202: [200, 200, 200],
        }
        layers = layer_cfg[depth]
        self.num_classes = num_classes
        self._norm_layer = nn.BatchNorm2D

        self.inplanes = 16
        self.dilation = 1

        self.conv1 = nn.Conv2D(
            3,
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, 1, 16,
                  previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.fc(x)
        return x


# Hyperparameters
BATCH_SIZE = 128

# Data augmentation and preprocessing
print('Downloading and loading training and testing data...')
transform_train = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(0.5),
    T.Transpose(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
])

transform_test = T.Compose([
    T.Transpose(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
])

train_data = paddle.vision.datasets.Cifar10(mode="train", transform=transform_train)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = paddle.vision.datasets.Cifar10(mode="test", transform=transform_test)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
print(f'Data loading complete. Training samples: {len(train_data)}, Test samples: {len(test_data)}.')

# Callback class for logging training and evaluation metrics
class AccLossCallback(paddle.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_train_acc = []
        self.epoch_train_loss = []
        self.epoch_eval_acc = []
        self.epoch_eval_loss = []

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_train_loss.append(logs.get('loss')[0])
        self.epoch_train_acc.append(logs.get('acc'))

    def on_eval_end(self, logs=None):
        self.epoch_eval_loss.append(logs.get('loss')[0])
        self.epoch_eval_acc.append(logs.get('acc'))


# Instantiate and configure the model
model = ResNet(BasicBlock, 32)
model = paddle.Model(model)

# Logging callback
mylogs = AccLossCallback()

# Learning rate scheduler and optimizer
scheduler = paddle.optimizer.lr.PiecewiseDecay(
    boundaries=[80, 120],
    values=[0.1, 0.01, 0.001],
    verbose=False
)
optim = paddle.optimizer.Momentum(
    parameters=model.parameters(),
    learning_rate=scheduler,
    momentum=0.9,
    weight_decay=1e-4
)

# Compile the model
model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    paddle.metric.Accuracy()
)

# Train the model
model.fit(train_loader, test_loader, epochs=150, batch_size=BATCH_SIZE,
          callbacks=[mylogs], verbose=1)

# Visualization of loss and accuracy
def plot_loss_acc(callback):
    plt.figure(figsize=(12, 6))
    plt.suptitle("Training and Evaluation Metrics")
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(callback.epoch_train_loss, label='Training Loss')
    plt.plot(callback.epoch_eval_loss, label='Evaluation Loss')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(callback.epoch_train_acc, label='Training Accuracy')
    plt.plot(callback.epoch_eval_acc, label='Evaluation Accuracy')
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_loss_acc(mylogs)