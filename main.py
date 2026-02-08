import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import random
from utils import plot_example, plot_example2

class SiameseDataset(Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels
        self.labels_to_indices = {}

        for idx, label in enumerate(labels):
            """
            The statement below uses setdefault to check if the current 'label' already exists, if it does
            returns the respective list, if it doesn't, creates a new label alongside an empty list, and returns
            the list. The append function takes the list and appends the current index to the end. The final
            library contains each type of label as a key, and a list of indices where these labels occur in the 
            data. 
            """
            self.labels_to_indices.setdefault(label,[]).append(idx)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        This function is ought to return a two data points(x1, x2) and a label (0 or 1) specifying
        whether these two are similar or not. idx is taken as the index of first data point and idx2 
        is the index of 2nd datapoint. idx2 is chosen based on whether a similar or a different second 
        datapoint is needed. 
        """
        x1 = self.data[idx]
        label1 = self.labels[idx]

        # print(label1)
        # plot_example2(x1)

        # Following randomly determines whether the current getitem returns a Similar or a Different 
        # tuple to be used. 
        same_class = random.choice([True, False])


        if same_class:
            idx2 = random.choice(self.labels_to_indices[label1])
            label = 1
            

        else:
            label2 = random.choice(list(set(self.labels_to_indices.keys()) - {label1}))
            idx2 = random.choice(self.labels_to_indices[label2])
            label = 0
            
        x2 = self.data[idx2]

        # Labels passed as a float object, as the loss functions are optimized for float32
        return x1.float().unsqueeze(0), x2.float().unsqueeze(0), torch.tensor(label, dtype= torch.float32)


class SiameseDatasetTrip(Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels
        self.labels_to_indices = {}

        for idx, label in enumerate(labels):
            """
            The statement below uses setdefault to check if the current 'label' already exists, if it does
            returns the respective list, if it doesn't, creates a new label alongside an empty list, and returns
            the list. The append function takes the list and appends the current index to the end. The final
            library contains each type of label as a key, and a list of indices where these labels occur in the 
            data. 
            """
            self.labels_to_indices.setdefault(label,[]).append(idx)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        In contrast to the other dataset, this dataset returns a triplet (Anchor, Positive, Negative). The positive
        is similar to the Anchor while the negative is not 
        """
        xAnc = self.data[idx]
        label1 = self.labels[idx]

        idxPos = random.choice(self.labels_to_indices[label1])  # Randomly chooses an index from the positives 
        xPos = self.data[idxPos]
        label2 = random.choice(list(set(self.labels_to_indices.keys()) - {label1})) # Randomly chooses a negative label
        idxNeg = random.choice(self.labels_to_indices[label2])                      # Randomly chooses an index from chosen label
        xNeg = self.data[idxNeg]

        # The Model requires the input to be in a specific form. The following includes another dimension turns data into float
        return xAnc.float().unsqueeze(0), xPos.float().unsqueeze(0),xNeg.float().unsqueeze(0)



class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.emb_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels= 32, kernel_size= 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2),
            nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size= 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2),
            nn.Flatten(),
            nn.LazyLinear(out_features= 256),
            nn.ReLU(),
            nn.Linear(256,100)
        )
        
    
    def forward(self, x):
        x = self.emb_net(x)
        
        return F.normalize(x, p= 2, dim= 1)


class SiameseNet(nn.Module):
    def __init__(self, emb_net):
        super().__init__()
        self.emb_net = emb_net

    def forward(self, x1, x2, x3 = torch.zeros(2, 1, 28, 28)):
        x3 = x3.to(x1.device)       # This makes sure that x3 is in the same device as x1. Dummy x3 made in CPU
        z1 = self.emb_net(x1)
        z2 = self.emb_net(x2)
        z3 = self.emb_net(x3)
        return z1, z2, z3


def train(train_dataloader, model, device, loss_fn, optim):
    dataset_size = len(train_dataloader.dataset)
    batch_size = train_dataloader.batch_size

    model.train()

    for batch, (xAnc, xPos, xNeg) in enumerate(train_dataloader):
        xAnc, xPos, xNeg = xAnc.to(device), xPos.to(device), xNeg.to(device)

        zAnc, zPos, zNeg = model(xAnc, xPos, xNeg)
        loss = loss_fn(zAnc, zPos, zNeg)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (batch%50) == 0:
            train_loss, current = loss.item(), batch * batch_size + len(xAnc)
            print(f"loss: {train_loss:>7f}  [{current:>5d}/{dataset_size:>5d}]")



def validate(val_dataloader, model, device):

    margin = 0.1
    total_correct = 0
    dataset_size = len(val_dataloader.dataset)

    model.eval()

    with torch.no_grad():
        for x1, x2, label in val_dataloader:
            
            
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)

            z1, z2, _ = model(x1, x2)
            similarity = (z1*z2).sum(dim=1)

            mask = similarity > (label - margin)
            total_correct += mask.sum().item()

            
    
    print(f"Accuracy :{total_correct}/{dataset_size} ")




if __name__ == '__main__':
        
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

    train_set_size = int(0.9*len(mnist_dataset))
    val_set_size = len(mnist_dataset) - train_set_size

    # Splits the MNIST dataset into train and val 
    train_set, val_set = random_split(mnist_dataset, [train_set_size, val_set_size])

    """
    Turns each MNIST subset into the requisite Siamese net datasets. train_set is a dataset containing 3-tuples
    with Anchor, Positive and Negative. val_set is a dataset containing 3-tuples, each with two randomly chosen examples
    and a 3rd element which is a label specifying whether a two elements are similar or not. 
    """
    train_set = SiameseDatasetTrip(train_set.dataset.data[train_set.indices], train_set.dataset.targets[train_set.indices].tolist())
    val_set = SiameseDataset(val_set.dataset.data[val_set.indices], val_set.dataset.targets[val_set.indices].tolist())


    triplet_train_dataloader = DataLoader(
        train_set,
        batch_size= 64,
        shuffle= True,
        num_workers= 5,
        pin_memory= True
    )

    val_dataloader = DataLoader(
        val_set,
        batch_size= 64,
        shuffle= False,
        num_workers= 5,
        pin_memory= True
    )

    epochs = 2

    model = SiameseNet(EmbeddingNet()).to(device)
    loss_func = nn.TripletMarginLoss(margin= 0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    

    for t in range(epochs):
                
        print(f"Running Epoch {t+1}")
        train(triplet_train_dataloader, model, device, loss_func, optimizer)
        validate(val_dataloader, model, device)

    torch.save(model.state_dict(), 'weights/SiamNet_weights.pth')
    print("Model saved!")



