import torch
import torchvision
import torchvision.transforms as transforms

from densenet import DenseNet121

PATH = "./cifar_net.pth"

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0,5, 0,5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 4

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)

classes = (
    "plane", "car",  "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)

net = DenseNet121()
net.load_state_dict(torch.load(PATH))

correct, total = 0, 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of hte network on the 10000 test images: {100 * correct // total}%')