import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy import signal

allowed_categories = {
    '14': 0,
    '11': 1,
    '21': 2,
    '6': 3,
    '23': 4
}
num_classes = len(allowed_categories)

class SimpleConvLayer:
    def __init__(self, input_shape, depth, kernel_size, learning_rate=0.01):
        self.input_depth, self.input_height, self.input_width = input_shape
        self.depth = depth
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.kernels = np.random.randn(depth, self.input_depth, kernel_size, kernel_size)
        self.biases = np.random.randn(depth, 1)

    def forward(self, input):
        self.input = input
        self.output_height = self.input_height - self.kernel_size + 1
        self.output_width = self.input_width - self.kernel_size + 1
        self.output = np.zeros((self.depth, self.output_height, self.output_width))
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        for i in range(self.depth):
            self.output[i] += self.biases[i]
        return self.output

    def backward(self, output_gradient):
        kernels_gradient = np.zeros_like(self.kernels)
        input_gradient = np.zeros_like(self.input)
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")
        self.kernels -= self.learning_rate * kernels_gradient
        self.biases -= self.learning_rate * output_gradient.sum(axis=(1, 2)).reshape(-1, 1)
        return input_gradient

class SimpleObjectDetector(nn.Module):
    def __init__(self, num_classes):
        super(SimpleObjectDetector, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2_class = nn.Linear(128, num_classes)
        self.fc2_bbox = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = self.dropout(F.relu(self.fc1(x)))
        class_scores = self.fc2_class(x)
        bbox = self.fc2_bbox(x)
        return class_scores, bbox

def combined_loss(pred_class_scores, true_classes, pred_bboxes, true_bboxes):
    valid_indices = (true_classes != -1).nonzero(as_tuple=True)[0]
    if valid_indices.numel() == 0:
        return torch.tensor(0.0, requires_grad=True)
    valid_pred_class_scores = pred_class_scores[valid_indices]
    valid_true_classes = true_classes[valid_indices]
    valid_pred_bboxes = pred_bboxes[valid_indices]
    valid_true_bboxes = true_bboxes[valid_indices]
    class_loss = F.cross_entropy(valid_pred_class_scores, valid_true_classes)
    bbox_loss = F.smooth_l1_loss(valid_pred_bboxes, valid_true_bboxes)
    return class_loss + bbox_loss

class FoodWasteDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        label_path = os.path.join(self.label_folder, os.path.splitext(self.image_files[idx])[0] + '.txt')

        boxes = []
        classes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    original_class_id = parts[0]
                    if original_class_id in allowed_categories:
                        class_id = allowed_categories[original_class_id]
                        x_center, y_center, width, height = map(float, parts[1:])
                        boxes.append([x_center, y_center, width, height])
                        classes.append(class_id)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.long)

        if len(classes) == 0:
            classes = torch.tensor([-1], dtype=torch.long)
            boxes = torch.zeros((1, 4), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, classes, boxes

def collate_fn(batch):
    images, classes, boxes = zip(*batch)
    images = torch.stack(images)
    max_boxes = max([b.shape[0] for b in boxes])
    padded_boxes = [torch.cat([b, torch.zeros(max_boxes - b.shape[0], 4)]) for b in boxes]
    padded_classes = [torch.cat([c, torch.full((max_boxes - c.shape[0],), -1)]) for c in classes]
    return images, torch.stack(padded_classes), torch.stack(padded_boxes)

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

image_folder = r'/home/niranjan/documents/projects/hack-umass-2024/ZeroByte/images'
label_folder = r'/home/niranjan/documents/projects/hack-umass-2024/ZeroByte/labels'
dataset = FoodWasteDataset(image_folder=image_folder, label_folder=label_folder, transform=transform)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

model = SimpleObjectDetector(num_classes=num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0
    batch_count = 0
    for images, classes, boxes in data_loader:
        batch_loss = 0
        for i in range(len(images)):
            image = images[i].unsqueeze(0)
            true_classes = classes[i]
            true_boxes = boxes[i]

            if len(true_classes) == 1 and true_classes[0] == -1:
                continue

            pred_class_scores, pred_bboxes = model(image)

            image_loss = 0
            for j in range(len(true_classes)):
                target_class = true_classes[j].unsqueeze(0)
                target_bbox = true_boxes[j].unsqueeze(0)

                pred_class_score = pred_class_scores[0].unsqueeze(0)
                pred_bbox = pred_bboxes[0].unsqueeze(0)

                object_loss = combined_loss(pred_class_score, target_class, pred_bbox, target_bbox)
                image_loss += object_loss

            image_loss = image_loss / len(true_classes)
            batch_loss += image_loss

        if batch_loss > 0:
            batch_loss = batch_loss / len(images)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            batch_count += 1

    if batch_count > 0:
        avg_loss = total_loss / batch_count
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

unlabeled_folder = r'/home/niranjan/documents/projects/hack-umass-2024/ZeroByte/train'
output_label_folder = r'/home/niranjan/documents/projects/hack-umass-2024/ZeroByte/new food waste annotations'
os.makedirs(output_label_folder, exist_ok=True)

model.eval()
confidence_threshold = 0.4

with torch.no_grad():
    for img_file in os.listdir(unlabeled_folder):
        if not (img_file.endswith('.jpg') or img_file.endswith('.png')):
            continue

        image_path = os.path.join(unlabeled_folder, img_file)
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        pred_class_scores, pred_bboxes = model(image_tensor)

        annotations = []
        for idx in range(pred_class_scores.size(0)):
            score_vector = pred_class_scores[idx]
            confidence, pred_class = torch.max(score_vector, dim=0)

            if idx < pred_bboxes.size(0):
                pred_bbox = pred_bboxes[idx].cpu().numpy()
                pred_bbox = np.clip(pred_bbox, 0, 1)

                if confidence.item() > confidence_threshold and pred_class.item() in allowed_categories.values():
                    annotations.append(f"{pred_class.item()} {pred_bbox[0]} {pred_bbox[1]} {pred_bbox[2]} {pred_bbox[3]}")

        annotation_path = os.path.join(output_label_folder, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(annotation_path, "w") as f:
            if annotations:
                f.write("\n".join(annotations))
            else:
                f.write("No valid detections")
                print(f"No valid detections for {img_file}")

print("Automatic annotation of 7,000 images completed.")
