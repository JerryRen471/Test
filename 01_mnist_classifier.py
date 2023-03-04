import torch
import torch.nn as nn
import pandas

#@title Classifier
class Classifier(nn.Module):

    def __init__(self):
        super().__init__() # 初始化父类

        # 定义神经网络层
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.Sigmoid(),
            nn.Linear(200, 10),
            nn.Sigmoid()
        )

        # 创建损失函数
        self.loss_function = nn.MSELoss()

        # 创建优化器，此处使用简单的随机梯度下降
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

        # 记录训练进展的计数器和列表
        self.counter = 0
        self.progress = []

    # 向网络传递信息
    def forward(self, inputs):
        # 直接运行模型
        return self.model(inputs)

    def train(self, inputs, targets):
        # 计算网络的输出值
        outputs = self.forward(inputs)
        # 计算损失值
        loss = self.loss_function(outputs, targets)
        # 梯度归零，反向传播，并更新权重
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())# item()方法是为了展开一个单值张量，获取里面的数字
            pass
        if (self.counter % 10000 == 0):
            print("counter = ", self.counter)
            pass
  
    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        pass

from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class MnistDataset(Dataset):
    def __init__(self, csv_file):
        self.data_df = pandas.read_csv(csv_file, header=None)
        pass

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        # 目标图像（标签）
        label = self.data_df.iloc[index, 0]
        target = torch.zeros((10))
        target[label] = 1.0

        # 图像数据，取值范围位0-255，标准化为0-1
        image_values = torch.FloatTensor(self.data_df.iloc[index, 1:].values) / 255

        # 返回标签、图像数据张量以及目标张量
        return label, image_values, target
    
    def plot_image(self, index):
        arr = self.data_df.iloc[index, 1:].values.reshape(28, 28)
        plt.title("label = " + str(self.data_df.iloc[index, 0]))
        plt.imshow(arr, interpolation='none', cmap='Blues')
        plt.show()
        pass

mnist_dataset = MnistDataset("D:\\Manual\\Code\\Data\\mnist_train.csv")
mnist_dataset.plot_image(9)


# 创建神经网络
C = Classifier()

# 在mnist数据集训练神经网络
epochs = 3

for i in range(epochs):
  print('training epoch', i+1, "of", epochs)
  for label, image_data_tensor, target_tensor in mnist_dataset:
    C.train(image_data_tensor, target_tensor)
    pass
  pass

# 绘制分类器损失值
C.plot_progress()