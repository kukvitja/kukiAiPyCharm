import torch

x = torch.rand([1,100,3])
target = torch.randn(1,100,64)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(3,64)

    def forward(self, x):
        x = self.fc1(x)
        return x



model = Net()
model(x)

par = list(model.parameters())
x.matmul(par[0].T) * par[1]

criterion = torch.nn.MSELoss()
optimize = torch.optim.SGD(model.parameters(), lr=0.0003)


for epoch in range(10):
    output = model(x)
    loss = criterion(output, target)
    print("E", epoch, "Loss", loss.item())
    optimize.zero_grad()
    loss.backward()
    optimize.step()


