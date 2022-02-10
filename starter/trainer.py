import torch
# had trouble with latest version of torchvision -- downgraded it. 
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
import torch.nn as nn

from densenet import DenseNet121

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0,5, 0,5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    # what is num_workers? I understand all the other parameters though
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)

"""
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)
"""

classes = (
    "plane", "car",  "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)

net = DenseNet121()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    # What do the parameters mean?
    net.parameters(), lr=0.001, momentum=0.9
)

def train(epoch):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/2000:.3f}')
            running_loss = 0.0

FINAL_EPOCH = 2


for epoch in range(FINAL_EPOCH):
    train(epoch)

print("Finished Training")

PATH = "./cifar_net.pth"
torch.save(net.state_dict(), PATH)
