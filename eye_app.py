# -*- coding: utf-8 -*-
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import zipfile  # Added for handling ZIP file extraction
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class definition
class EyeDiagramDataset(Dataset):
    def __init__(self, image_files, transform=None):  # Modified to accept file list instead of directory
        self.image_files = image_files
        self.transform = transform
        self.classes = ['crosstalk', 'Loss and ISI', 'reflection']
        self.labels = [self.get_label_from_filename(f) for f in self.image_files]  # Assign labels based on file names

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_label_from_filename(self, filename):  # This function extracts the class label from the filename
        # Modify this mapping logic based on how your filenames are structured
        if "crosstalk" in filename:
            return 0
        elif "Loss and ISI" in filename:
            return 1
        elif "reflection" in filename:
            return 2
        else:
            raise ValueError(f"Unrecognized class in file: {filename}")

# Data transformation definition
data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the model (ResNet101 with Transfer Learning)
def get_model(num_classes):
    model = models.resnet101(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, num_classes)
    )

    return model

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=100):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if scheduler:
            scheduler.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    return train_losses, train_accs, val_losses, val_accs

# Confusion Matrix function
def get_confusion_matrix(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return confusion_matrix(all_labels, all_preds)

# Learning curves plot function
def plot_learning_curves(train_losses, train_accs, val_losses, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss Curves')

    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Accuracy Curves')

    plt.tight_layout()
    return fig

# Image prediction function
def predict_image(model, image, class_names):
    model.eval()
    image = image.to(device)  # Ensure 4D input [1, 3, 224, 224]

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()

    return predicted_class, probabilities

# Streamlit app
def main():
    st.title("Eye Diagram Classifier (ZIP Upload)")

    # Upload ZIP files instead of directory paths
    train_zip = st.file_uploader("Upload training data (ZIP file)", type="zip")  # Modified to handle ZIP file upload
    val_zip = st.file_uploader("Upload validation data (ZIP file)", type="zip")  # Modified to handle ZIP file upload

    if train_zip and val_zip:
        # Extract training ZIP
        with zipfile.ZipFile(train_zip, 'r') as zip_ref:
            zip_ref.extractall("/tmp/train_data")
        train_files = [os.path.join("/tmp/train_data", f) for f in os.listdir("/tmp/train_data") if f.endswith(('jpg', 'png', 'jpeg'))]

        # Extract validation ZIP
        with zipfile.ZipFile(val_zip, 'r') as zip_ref:
            zip_ref.extractall("/tmp/val_data")
        val_files = [os.path.join("/tmp/val_data", f) for f in os.listdir("/tmp/val_data") if f.endswith(('jpg', 'png', 'jpeg'))]

        # Create datasets and dataloaders
        train_dataset = EyeDiagramDataset(train_files, transform=data_transform)
        val_dataset = EyeDiagramDataset(val_files, transform=data_transform)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        num_classes = len(train_dataset.classes)

        # Model saving path (use /tmp for Streamlit Cloud)
        model_name = f'/tmp/eye_diagram_classifier_v1.pth'

        # Class weights for imbalanced dataset
        class_weights = torch.FloatTensor([0.7, 1.0, 1.2]).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        # Train the model
        if st.button("Train Model"):
            model = get_model(num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

            train_losses, train_accs, val_losses, val_accs = train_model(
                model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100
            )

            # Plot learning curves
            fig = plot_learning_curves(train_losses, train_accs, val_losses, val_accs)
            st.pyplot(fig)

            # Confusion Matrix
            cm = get_confusion_matrix(model, val_loader)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title('Confusion Matrix')
            ax.xaxis.set_ticklabels(train_dataset.classes)
            ax.yaxis.set_ticklabels(train_dataset.classes)
            st.pyplot(fig)

            # Save the trained model
            torch.save(model.state_dict(), model_name)
            st.success(f"Model trained and saved successfully as {model_name}!")

        # Model testing
        st.subheader("Model Testing")
        uploaded_file = st.file_uploader("Upload an eye diagram image for testing", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Eye Diagram', use_column_width=True)

            model = get_model(num_classes).to(device)
            model.load_state_dict(torch.load(model_name))
            model.eval()

            # Preprocess the image
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = test_transform(image).unsqueeze(0).to(device)

            # Make predictions
            predicted_class, probabilities = predict_image(model, input_tensor, train_dataset.classes)

            st.write(f"Predicted Class: {train_dataset.classes[predicted_class]}")
            for i, prob in enumerate(probabilities):
                st.write(f"{train_dataset.classes[i]}: {prob:.4f}")

if __name__ == "__main__":
    main()
