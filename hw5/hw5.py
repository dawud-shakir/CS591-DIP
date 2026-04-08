# hw5.py

# Hyperparameters

batch_size = 64
learning_rate = 0.001
epochs = 10

max_items = 200   # per category
train_size, test_size = 0.5, 0.5

device = 'cpu'
        

from pathlib import Path

data_dir = Path(__file__).parent / "data"

import numpy as np
import matplotlib.pyplot as plt
import os
import time

import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import urllib.request

# Define categories to download
# See all the categories at:
# https://github.com/googlecreativelab/quickdraw-dataset/blob/master/categories.txt

base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

# 5 classes
categories = [
    "cat",
    "fish",
    "flower",
    "star",
    "tree"
]

for cat in categories:
    url = f"{base_url}{cat.replace(' ', '%20')}.npy"    # replace spaces with %20
    path = data_dir / f"{cat}.npy"
    if not os.path.exists(path):
        print(f"Downloading {cat}...")
        urllib.request.urlretrieve(url, path)

#################################################################

class QuickDrawDataset(Dataset):
    def __init__(self, root, categories, max_items=10000, transform=None):
        super().__init__()
        self.data, self.labels = [], []
        self.transform = transform

        for idx, cat in enumerate(categories):
            # Load as raw uint8 (0 - 255)
            data = np.load(f"{root}/{cat}.npy")[:max_items]
            
            self.data.append(data)
            self.labels.extend([idx] * len(data))

        self.data = np.concatenate(self.data).astype(np.uint8)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].reshape(28, 28)
        label = self.labels[idx]

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        return image, label

################################################################

# Setup

raw_dataset = QuickDrawDataset(root=data_dir, categories=categories, max_items=max_items)
loader = DataLoader(raw_dataset, batch_size=batch_size)

train_dataset, test_dataset = random_split(raw_dataset, [train_size, test_size])

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False,
)    

""" TOFIX: this only finds the mean/std of a single batch """
all_images, all_labels = next(iter(loader))

all_images = all_images.float()
all_mean, all_std = all_images.mean().item(), all_images.std().item()
all_mean, all_std = all_mean / 255.0, all_std / 255.0

raw_dataset.transform = transforms.Compose([
    # [0, 255] -> [0.0, 1.0]
    transforms.ToTensor(),
    transforms.Normalize((all_mean,), (all_std,))
])

models = {
    "config_1": {
        "desc": "Baseline NN",
        "model": nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(categories)),
        ),
    },

    
    "config_2": {
        "desc": "CNN-A",
        "model": nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*14*14, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(categories)),
        ),
        
    },
    "config_3": {
        "desc": "CNN-B",
        "model":nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*14*14, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(categories)),
        ),
    },
    "config_4": {
        "desc": "CNN-C",
        "model":nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(categories)),
        ),
    },
}

# Variable-layer neural network for classification
class Net(nn.Module):
    def __init__(self, layers: nn.Sequential):
        super(Net, self).__init__()
        self.layers = layers

    def forward(self, x):
        return self.layers(x)

def evaluate(model, loader):
    model.eval()

    correct, total = 0, 0
    timer = time.perf_counter()

    with torch.no_grad():
        for (data, target) in loader:
            output = model(data)

            pred_idx = output.argmax(dim=1)

            correct += (pred_idx == target).sum().item()
            total += target.size(0)

    return correct / total, time.perf_counter() - timer

def predict(model, img_gray):
        # Resize
        img_28 = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)

        # Match the QuickDraw format (0 to 1 range)
        img_tensor = torch.from_numpy(img_28).float() / 255.0

        # Normalize by our selected dataset
        img_tensor = (img_tensor - all_mean) / all_std
        
        # Normalize by entire Quickdraw dataset
        # img_tensor = (img_tensor - 0.5) / 0.5
        
        # [1, 1, 28, 28]
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)

        # Predict
        model.eval()
        with torch.no_grad():
            output_logits = model(img_tensor)
            ps = torch.softmax(output_logits, dim=1)
            pred_conf, pred_idx = torch.max(ps, dim=1)

        # Visual Result
        
        # Map index back to class name
        pred_idx = int(pred_idx.item())
        pred_label = categories[pred_idx] if 'categories' in globals() else pred_idx

        plt.figure(figsize=(3, 4))
        plt.imshow(img_28, cmap='gray')
        plt.title(f"Result: {pred_label}\nConf: {pred_conf.item()*100:.1f}%")
        plt.axis('off')
        plt.show()

        return pred_label, pred_conf.item()

def epoch_canvas_predictions(model):
        img_labels = ["cat", "fish", "flower", "star", "tree",]
        
        what_pred_conf = []

        for i, img_label in enumerate(img_labels):
            img_gray = cv2.imread(data_dir / f"{img_label}.png", cv2.IMREAD_GRAYSCALE) 
            # Resize and Normalize
            img_28 = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)

            # Match the QuickDraw format (0 to 1 range)
            img_tensor = torch.from_numpy(img_28).float() / 255.0
            # normalization
            img_tensor = (img_tensor - all_mean) / all_std
            # img_tensor = (img_tensor - 0.5) / 0.5
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)

            # Prediction
            model.eval()
            with torch.no_grad():
                output_logits = model(img_tensor)
                ps = torch.softmax(output_logits, dim=1)
                pred_conf, pred_idx = torch.max(ps, dim=1)

            what_pred_conf.append((img_label, categories[pred_idx.item()], pred_conf.item()))

        return what_pred_conf

def show_canvas_predictions(what_pred_conf):
    rows = len(what_pred_conf) // len(categories)   # each epoch has one set of categories
    cols = len(categories)
    fig, axs = plt.subplots(rows, cols)
    fig.set_facecolor('black')
    for i, ax in enumerate(axs.flat):
        img_gray = cv2.imread(data_dir / f"{what_pred_conf[i][0]}.png", cv2.IMREAD_GRAYSCALE) 
        img_28 = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
        img_28 = img_28.astype(float) / 255.0
        img_28 = (img_28 - all_mean) / all_std
        # img_28 = (img_28 - 0.5) / 0.5

        ax.imshow(img_28, cmap='gray')

        true_label = what_pred_conf[i][0]
        pred_label = what_pred_conf[i][1]
        confidence = what_pred_conf[i][2]
        title = f"[Pred: {pred_label} Conf: {confidence:.2f}%]"

        ax.set_title(title, fontsize=6, color='white')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.subplots_adjust(wspace=0.5, hspace=1.0, bottom=-1.0)    
    plt.show()

def run(config):
    model = Net(layers=models[config]["model"])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()

    # Training
    losses = []
    accuracies = []



    print('\n##', models[config]["desc"])
    what_pred_conf = []

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0

        train_time = time.perf_counter()

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)

            if isinstance(criterion, nn.NLLLoss):
                # NLLLoss uses log probabilities
                output = torch.log_softmax(output, dim=1)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                losses.append(loss.item())

                # pred_label, confidence = test_canvas_prediction(model, img_1)
                # print(f"Epoch {epoch} says a {img_class} is a {pred_label} with {confidence:.2f} confidence!")

            # Training accuracy
            pred_idx = output.argmax(dim=1)
            correct += (pred_idx == target).sum().item()
            total += target.size(0)

        train_acc = correct / total
        train_time = time.perf_counter() - train_time

        val_acc, val_time = evaluate(model, test_loader)

        accuracies.append(val_acc)

        # what_pred_conf.extend(epoch_canvas_predictions(model))

    # show_canvas_predictions(what_pred_conf)

    print(f"Epoch {epoch+1} complete.")

    # overall variability across the whole training run
    stability = np.std(losses)
    
    print(f"Train/Test Acc: {train_acc:.2f}/{accuracies[-1]:.2f}")
    print(f"Train/Test Time: {train_time:.3f}/{val_time:.3f}")
    print(f"Stability: {stability:.3f}")




    images, labels = next(iter(train_loader))
    for i in range(len(images)):

        img = images[i].squeeze(0).cpu().numpy().astype(np.uint8)
        ret = predict(model, img) 

        print(f"True label: {categories[labels[i].item()]}")
        print(ret)

        pass        

    return losses, accuracies





if __name__ == "__main__":

    for config in models:
        losses, accuracies = run(config=config)


    # _, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].plot(losses)
    # axs[0].set_title("Training Loss")
    # axs[0].set_xlabel("Iterations (per 100 batches)")
    # axs[0].set_ylabel("Loss")
    # axs[0].grid(True)

    # axs[1].plot(accuracies)
    # axs[1].set_title("Test Accuracy")
    # axs[1].set_xlabel("Epoch")
    # axs[1].set_ylabel("Accuracy")
    # axs[1].grid(True)

    # plt.show()