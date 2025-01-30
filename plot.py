import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)

# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# Data Augmentation and normalization
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(root='C:/Users/HP STAND BY LAPTOP/Desktop/Major_project/dataset/training_data', transform=transform)
test_dataset = datasets.ImageFolder(root='C:/Users/HP STAND BY LAPTOP/Desktop/Major_project/dataset/test_data', transform=transform)

# Split the dataset into training and validation sets
total_size = len(train_dataset)
train_size = int(0.7 * total_size)
val_size = total_size - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # Ensures output values are between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the Autoencoder model
autoencoder_model = Autoencoder()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder_model.parameters(), lr=0.001)

# Training loop
train_losses = []
val_losses = []
val_accuracies = []
all_predictions = []
all_labels = []

for epoch in range(EPOCHS):
    autoencoder_model.train()
    total_loss = 0.0
    for inputs, _ in train_loader:
        optimizer.zero_grad()
        outputs = autoencoder_model(inputs)
        loss = criterion(outputs, inputs)  # Mean Squared Error loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    train_losses.append(average_loss)
    print(f'Epoch {epoch + 1}/{EPOCHS}, Training Loss: {average_loss:.4f}')

    # Validation loop
    autoencoder_model.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, _ in val_loader:
            outputs = autoencoder_model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()

            # For accuracy calculation
            predictions = (outputs > 0.5).float().cpu().numpy()
            labels = (inputs > 0.5).float().cpu().numpy()

            # Convert to binary predictions
            binary_predictions = (predictions > 0.5).astype(int)
            binary_labels = (labels > 0.5).astype(int)

            # Flatten the arrays
            all_predictions.extend(binary_predictions.reshape(-1))
            all_labels.extend(binary_labels.reshape(-1))

        average_loss = total_loss / len(val_loader)
        val_losses.append(average_loss)
        accuracy = accuracy_score(all_labels, all_predictions)
        val_accuracies.append(accuracy)
        print(f'Epoch {epoch + 1}/{EPOCHS}, Validation Loss: {average_loss:.4f}, Validation Accuracy: {accuracy:.4f}')

# Test the Autoencoder model
autoencoder_model.eval()
with torch.no_grad():
    total_loss = 0
    for inputs, _ in test_loader:
        outputs = autoencoder_model(inputs)
        loss = criterion(outputs, inputs)
        total_loss += loss.item()

        # For accuracy calculation
        predictions = (outputs > 0.5).float().cpu().numpy()
        labels = (inputs > 0.5).float().cpu().numpy()

        # Convert to binary predictions
        binary_predictions = (predictions > 0.5).astype(int)
        binary_labels = (labels > 0.5).astype(int)

        # Flatten the arrays
        all_predictions.extend(binary_predictions.reshape(-1))
        all_labels.extend(binary_labels.reshape(-1))

    test_loss = total_loss / len(test_loader)
    test_accuracy = accuracy_score(all_labels, all_predictions)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

def plot_curves(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 4))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

# Plot the Loss and Accuracy Curves
plot_curves(train_losses, val_losses, val_accuracies)

# Plot the Confusion Matrix
classes = ['Normal', 'Anomaly']
plot_confusion_matrix(all_labels, all_predictions, classes)

# Save the trained Autoencoder model
torch.save(autoencoder_model.state_dict(), 'incepKNN.pth')
