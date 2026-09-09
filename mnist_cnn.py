import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.ToTensor()

train_data = MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)
test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)

class SimpleCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.classifier = nn.Sequential(
            nn.Linear(16*7*7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# create model

model = SimpleCNNModel().to(device)

#loss function
loss_fn = nn.CrossEntropyLoss()

#optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 3

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} |  Loss {total_loss:.4f}")

model.eval()
total = 0
correct = 0

with torch.no_grad():
    for images, labels in test_loader:  # evaluate on the held-out test set
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        prediction = outputs.argmax(dim=1)

        correct += (prediction == labels).sum().item()

        total += labels.size(0)

    accuracy = 100 * correct / total

    print(f"Test accuracy {accuracy:.2f}%")
