import torch
from torchvision import transforms
from PIL import Image
import numpy as np

model = SimpleObjectDetector(num_classes=num_classes)
model.load_state_dict(torch.load(r"C:\Users\Ahmad\PycharmProjects\zero-byte\ZeroByte\cnn.py"))
model.eval()

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

confidence_threshold = 0.5

def process_single_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        pred_class_scores, pred_bboxes = model(input_tensor)

    annotations = []
    for idx in range(pred_class_scores.size(0)):
        score_vector = pred_class_scores[idx]
        confidence, pred_class = torch.max(score_vector, dim=0)

        if idx < pred_bboxes.size(0) and confidence.item() > confidence_threshold:
            pred_bbox = pred_bboxes[idx].cpu().numpy()
            pred_bbox = np.clip(pred_bbox, 0, 1)
            annotations.append((pred_class.item(), confidence.item(), pred_bbox))

    return annotations
