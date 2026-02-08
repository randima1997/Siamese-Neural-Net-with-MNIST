import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import random
from utils import plot_example, plot_example2, plot_example3
from main import SiameseDataset, SiameseNet, EmbeddingNet

weights_path = '/home/randima/Projects/Siamese Neural Net with MNIST/weights/SiamNet_weights_3.pth'

device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

print(f"Using {device} device")

transform = transforms.Compose([
        transforms.ToTensor()
    ])

mnist_dataset = datasets.MNIST(
    root= '/home/randima/Projects/Siamese Neural Net with MNIST/MNIST',
    train= True,
    download= True,
    transform= transform
)


train_set_size = int(0.95*len(mnist_dataset))
val_set_size = len(mnist_dataset) - train_set_size

test_set, _ = random_split(mnist_dataset, [train_set_size, val_set_size])

test_set = SiameseDataset(test_set.dataset.data[test_set.indices], test_set.dataset.targets[test_set.indices].tolist())
# test_set = SiameseDataset(mnist_dataset, mnist_dataset.targets.tolist())


test_dataloader = DataLoader(
    test_set,
    batch_size= 32,
    shuffle= True,
    num_workers= 1,
    pin_memory= False
)

# h1, h2, lab = test_set[0]

model = SiameseNet(EmbeddingNet()).to(device)
model.load_state_dict(torch.load(weights_path, weights_only=True))
model = model.to(device)



for x1, x2, _ in test_dataloader:
    x1, x2 = x1.to(device), x2.to(device)
    z1, z2, _ = model(x1, x2)
    similarity = (z1*z2).sum(dim=1)
    
    for im1, im2, sim in zip(x1, x2, similarity):
        if sim.to('cpu').item() > 0.97:
            similarness = 'Similar'
        else:
            similarness = 'Not!'
        print("Similarity : ", similarness, "\t",sim.to('cpu').item())
        plot_example3(im1.to('cpu').squeeze(0), im2.to('cpu').squeeze(0))



# for h1, h2, lab in test_set:
#     print(lab.item())
#     plot_example3(h1.squeeze(0), h2.squeeze(0))
