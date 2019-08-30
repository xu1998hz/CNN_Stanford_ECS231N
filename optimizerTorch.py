import torch
from torch.autograd import Variable

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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
    # Forward pass: feed data to model, and prediction to loss function
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    # backward pass: compute all gradients
    model.zero_grad()
    loss.backward()
    # Update all parameters after computing gradients
    optimizer.step()
