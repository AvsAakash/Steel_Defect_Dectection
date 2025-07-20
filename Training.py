import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import torchvision.transforms as T
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET

# Define the custom dataset
class SteelDefectsDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for steel defect detection with XML annotations.
    """
    def __init__(self, images_dir, annotations_dir, transforms=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.image_files = sorted(
            [os.path.join(root, file)
             for root, _, files in os.walk(images_dir)
             for file in files if file.endswith(('.jpg', '.png'))]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        annotation_file = os.path.join(
            self.annotations_dir,
            os.path.relpath(img_path, start=self.images_dir).replace('.jpg', '.xml')
        )
        boxes, labels = self.parse_xml(annotation_file)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels}

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def parse_xml(self, annotation_file):
        tree = ET.parse(annotation_file)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            label = obj.find('name').text
            labels.append(self.label_to_int(label))

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

        return boxes, labels

    @staticmethod
    def label_to_int(label):
        label_mapping = {
            "scratches": 1,
        }
        return label_mapping.get(label, 0)

# Paths
weights_path = r"C:\Users\srika\.cache\torch\hub\checkpoints\fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
images_dir = r"C:\Users\srika\Downloads\NEU-DET-Steel-Surface-Defect-Detection-master\NEU-DET-Steel-Surface-Defect-Detection-master\IMAGES\Scratches"
annotations_dir = r"C:\Users\srika\Downloads\NEU-DET-Steel-Surface-Defect-Detection-master\NEU-DET-Steel-Surface-Defect-Detection-master\ANNOTATIONS\Scratches"

# Load Faster R-CNN model
weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn(weights=weights)

# Reinitialize the box predictor
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

# Dataset preparation
dataset = SteelDefectsDataset(
    images_dir=images_dir,
    annotations_dir=annotations_dir,
    transforms=T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define collate function
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

# Train the model
def train_model(model, dataset, device, num_epochs=15):
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    # Set deterministic behavior for reproducibility
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for epoch in range(num_epochs):
        total_loss = 0
        print(f"Epoch {epoch + 1}/{num_epochs}")
        progress_bar = tqdm(data_loader, desc=f"Training Epoch {epoch + 1}")

        model.train()
        for images, targets in progress_bar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            scaler.scale(losses).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()

            total_loss += losses.item()
            progress_bar.set_postfix(loss=losses.item())

        lr_scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

    print("Training completed!")
    torch.save(model.state_dict(), 'C:/Users/gks02/Downloads/SDD/Gradient_Clipping_faster_rcnn_steel_defects_OS15.pth')
    print("Model saved as Gradient_Clipping_faster_rcnn_steel_defects_OS15.pth")

if __name__ == "__main__":
    train_model(model, dataset, device)
