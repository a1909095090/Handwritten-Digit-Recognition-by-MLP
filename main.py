from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mlp import Mlp
import  torch

if __name__ == '__main__':
    epochs = 4  # Number of epochs
    reg = 1e-5  # 正则化参数
    optm = "Adam" # 优化器
    # optm = "SGD"
    # optm == "RMSprop"
    #Step1： 加载数据
    dataset = datasets.MNIST(root="./MNIST",
                            train=True,
                            transform=transforms.ToTensor(),
                            download=False)
    # Step 2: 计算前80%的数据点数量
    total_data_points = len(dataset)
    train_data_points = int(total_data_points * 0.8)

# Step 3: 选择前80%的数据作为训练集
    train_dataset = torch.utils.data.Subset(dataset, list(range(train_data_points)))
    test_dataset = torch.utils.data.Subset(dataset, list(range(train_data_points, total_data_points)))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# 选择损失函数和优化器:
    model =Mlp(opt=optm,reg=reg)
#Step 4 训练并且测试

    for epoch in range(1, epochs + 1):
        model.fit( train_loader, epoch)
        print(f'Epoch {epoch} finished')
        model.test(test_loader)
#Step 5： 保存模型:
    torch.save(model.state_dict(), 'mnist_mlp_model.pth')

