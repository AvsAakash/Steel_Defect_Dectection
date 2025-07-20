import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms
import cv2
from matplotlib import pyplot as plt

# Model Creation with ResNet-50 Backbone and FPN
def create_fasterrcnn_model(num_classes):
    """
    Create a Faster R-CNN model with ResNet-50, FPN, and updated box predictor.

    Parameters:
        num_classes (int): Number of classes including the background.

    Returns:
        model: Configured Faster R-CNN model.
    """
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Reconfigure the box predictor for custom classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# Preprocess the input image
def preprocess_image(image_path):
    """
    Load and preprocess an image for inference.

    Parameters:
        image_path (str): Path to the image.

    Returns:
        original_image: The original image in OpenCV format.
        transformed_image: The image after transformations.
    """
    image = cv2.imread(image_path)
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transformed_image = transform(original_image)
    return original_image, transformed_image.unsqueeze(0)

# Perform inference and visualize
def predict_and_visualize(model, image_tensor, device, original_image, score_threshold=0.3):
    """
    Perform prediction on an image and visualize the results.

    Parameters:
        model: Trained Faster R-CNN model.
        image_tensor: Transformed image tensor.
        device: PyTorch device.
        original_image: Original image for visualization.
        score_threshold (float): Minimum confidence score for visualization.

    Returns:
        None
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

    for box, score, label in zip(outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels']):
        if score >= score_threshold:
            x_min, y_min, x_max, y_max = map(int, box.cpu().numpy())
            cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            cv2.putText(
                original_image,
                f"Score : {score:.2f}",
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

    plt.imshow(original_image)
    plt.axis("off")
    plt.show()

# Main execution
def main():
    """
    Main function to run Scheme 4.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model and paths
    num_classes = 2  # 1 class + background
    model = create_fasterrcnn_model(num_classes)
    model.load_state_dict(torch.load(r"C:\Users\srika\OneDrive\Desktop\Sem\Code\project\Gradient_Clipping_faster_rcnn_steel_defects_OS15.pth"))
    model.to(device)

    image_path = r"C:\Users\srika\Downloads\NEU-DET-Steel-Surface-Defect-Detection-master\NEU-DET-Steel-Surface-Defect-Detection-master\IMAGES\Scratches\scratches_27.jpg"
    original_image, image_tensor = preprocess_image(image_path)

    predict_and_visualize(model, image_tensor, device, original_image)

if __name__ == "__main__":
    main()
