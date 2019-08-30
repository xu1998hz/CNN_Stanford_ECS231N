import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

class TwoLayerNet(torch.nn.Module):
    def _init_(self, D_in, H, D_out):
        super(TwoLayerNet, self)._init_()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

   def forward(self, x):
       h_relu = self.linear1(x).clamp(min=0)
       y_pred = self.linear2(h_relu)
       return y_pred

M, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

loader = DataLoader(TensorDataset(x, y), batch_size=8)

model = TwoLayerNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for epoch in range(10):
    for x_batch, y_batch in loader:
        x_var, y_var = Variable(x), Variable(y)
        y_pred = model(x_var)
        loss = criterion(y_pred, y_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
