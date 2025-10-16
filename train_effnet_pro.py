import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# =============================
# 1. Dataset & Augmentations
# =============================
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

data_dir = "eff_dataset"
image_datasets = {x: datasets.ImageFolder(root=f"{data_dir}/{x}",
                                          transform=data_transforms[x])
                  for x in ["train", "val", "test"]}

dataloaders = {x: DataLoader(image_datasets[x],
                             batch_size=32,
                             shuffle=True,
                             num_workers=2)
               for x in ["train", "val", "test"]}

class_names = image_datasets["train"].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\n✅ Using device: {device}\n")

# =============================
# 2. Load Model
# =============================
model = models.efficientnet_b2(weights="IMAGENET1K_V1")
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())  # AMP only if CUDA

# Ensure output folder exists
os.makedirs("efficientnet", exist_ok=True)

# =============================
# 3. Training Loop with Early Stopping
# =============================
def train_model(model, criterion, optimizer, scheduler, num_epochs=30, patience=5):
    best_acc = 0.0
    best_weights = None
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Save best weights
            if phase == "val":
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_weights = model.state_dict()
                    patience_counter = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': best_weights,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_acc': best_acc
                    }, "efficientnet/effnet_checkpoint.pth")
                    print("✅ Saved new best model checkpoint")
                else:
                    patience_counter += 1

        scheduler.step()

        # Early stopping
        if patience_counter >= patience:
            print("⛔ Early stopping triggered.")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    print(f"\n🏆 Best val Acc: {best_acc:.4f}")
    torch.save(model.state_dict(), "efficientnet/effnet_b2.pth")
    print("✅ Final model saved at efficientnet/effnet_b2.pth")
    return model

# =============================
# 4. Evaluation on Test Set
# =============================
def evaluate_model(model):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\n📌 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

# =============================
# 5. Export to ONNX
# =============================
def export_model(model):
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(model, dummy_input, "efficientnet/effnet_b2.onnx",
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"},
                                    "output": {0: "batch_size"}})
    print("\n✅ Model exported to efficientnet/effnet_b2.onnx")

# =============================
# Run Training
# =============================
if __name__ == "__main__":
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=10, patience=5)
    evaluate_model(model)
    export_model(model)
