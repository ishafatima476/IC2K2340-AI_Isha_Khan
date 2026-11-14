import torch
import torch.nn as nn
import torch.optim as optim

# Training data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Create neural network
model = nn.Linear(1, 1)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(1000):

    # Forward pass
    output = model(X)

    # Calculate error
    loss = criterion(output, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Update weights
    optimizer.step()

# Test prediction
test = torch.tensor([[5.0]])
prediction = model(test)

print("Prediction:", prediction.item())
