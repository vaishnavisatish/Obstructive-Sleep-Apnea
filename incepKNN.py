import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

IMAGE_SIZE = (299, 299)  
BATCH_SIZE = 32
EPOCHS = 15
K_NEIGHBORS = 7


transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])



train_dataset = datasets.ImageFolder(root='C:/Users/hp/Desktop/Major_project/dataset/training_data', transform=transform)
test_dataset = datasets.ImageFolder(root='C:/Users/hp/Desktop/Major_project/dataset/test_data', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class InceptionModel(nn.Module):
    def __init__(self, num_classes):
        super(InceptionModel, self).__init__()
        self.inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        num_ftrs = self.inception.fc.in_features
        self.inception.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.inception(x)

num_classes = len(train_dataset.classes)
model = InceptionModel(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        logits = outputs.logits  
        loss = criterion(logits, labels)  
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{EPOCHS}, Training Loss: {average_loss:.4f}')

model.eval()
test_features = []
test_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        test_features.extend(outputs.cpu().numpy())
        test_labels.extend(labels.numpy())

knn_classifier = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
knn_classifier.fit(test_features, test_labels)

test_features = []
test_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        test_features.extend(outputs.cpu().numpy())
        test_labels.extend(labels.numpy())

predictions = knn_classifier.predict(test_features)
accuracy = accuracy_score(test_labels, predictions)
print(f'Test Accuracy: {accuracy:.4f}')
