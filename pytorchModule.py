"""A PyTorch Module is a neural net layer; it inputs and outputs Variables
   Modules can contain weights(as Variables) or other Modules, define my own Modules using autograd"""

 import torch
 from torch.autograd import Variable

 class TwoLayerNet(torch.nn.Module):
     def _init_(self, D_in, H, D_out):
         super(TwoLayerNet, self)._init_()
         self.linear1 = torch.nn.Linear(D_in, H)
         self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 0

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

model = TwoLayerNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for t in range(500):
    # Forward pass: feed data to model, and prediction to loss function
    y_pred = model(x)
    loss = criterion(y_pred, y)
    # backward pass: compute all gradients
    model.zero_grad()
    loss.backward()
    # Update all parameters after computing gradients
    optimizer.step()
