import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class Mlp(nn.Module):
    def __init__(self,opt="Adam",reg=0.001):
        super(Mlp,self).__init__()

        self.mlp_net=nn.Sequential(
            nn.Linear(28*28,64,bias=False),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 10, bias=False),
        )
        self.reg=reg
        self.loss_fn=F.cross_entropy
        self.total_loss=0
        self.opt=opt

        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp_net.to(self.device)
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image
        x = self.mlp_net(x)
        return x

    def fit(self,  train_loader):
        criterion = self.loss_fn
        model= self.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        if self.opt == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=0.001)
        elif self.opt == "RMSprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            # 加上正则项
            loss = loss + self.reg*sum(p.sum() for p in model.parameters())
            loss.backward()
            optimizer.step()

    def test(self,  test_loader):
        criterion = self.loss_fn
        model=self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print(
                f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

