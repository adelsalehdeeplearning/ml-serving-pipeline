# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
iris = load_iris()
X, y = iris.data, iris.target
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# Define model
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train model
for epoch in range(100):
    optimizer.zero_grad()
    output = model(X_train)
    loss = loss_fn(output, y_train)
    print('loss:', loss.item())
    loss.backward()
    optimizer.step()

print("Training complete. Saving model...")

# Save model and scaler
torch.save(model.state_dict(), "model.pt")
torch.save(scaler, "scaler.pt")
