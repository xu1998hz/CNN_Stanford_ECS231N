import torch
from torch.autograd import Variable

# define my own autograd functions by writing forward and backward for Tensors
class ReLU(torch.autograd.Function):
    def forward(self, x):
        self.save_for_backward(x)
        return x.claemp(min=0)

    def backward(self, grad_y):
        x, = self.saved_tensors
        grad_input = grad_y.clone()
        grad_input[x < 0] = 0
        return grad_input

N, D_in, H, D_out = 64, 1000, 100, 10
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad = False)

# Define our model as a sequence of layers
model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out)
)
# nn also defines common loss function
loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4


for t in range(500):
    # Forward pass: feed data to model, and prediction to loss function
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    # backward pass: compute all gradients
    model.zero_grad()
    loss.backward()
    # Make gradient step on each model parameter
    for param in model.parameters():
        param.data -= learning_rate * param.grad,data
