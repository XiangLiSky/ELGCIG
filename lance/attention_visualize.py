import cv2
import numpy as np
import torch
import os
from torchvision.models import resnet50, ResNet50_Weights
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(torch.device("cuda")).eval()
# model.load_state_dict(torch.load('../output_enhance_addi/baseball_retrained_fused.pth'))
target_layers = [model.layer4[-1]]
rgb_img = cv2.imread('../additional_exp/val/baseball/ILSVRC2012_val_00008099.JPEG', 1)[:, :, ::-1]
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).to(torch.device("cuda"))
# We have to specify the target we want to generate the CAM for.

categories = []
with open("../imagenet_classes.txt", "r") as f:
    for line in f:
        parts = line.strip().split(', ')
        if len(parts) == 2:
            categories.append(parts[1])
        else:
            print(f"Unexpected format in line: {line}")

actual_class_index = categories.index("baseball")

targets = [ClassifierOutputTarget(actual_class_index)]

with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
  grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
  grayscale_cam = grayscale_cam[0, :]
  cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
  cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)


gb_model = GuidedBackpropReLUModel(model=model, device=torch.device("cuda"))
gb = gb_model(input_tensor, target_category=None)

cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
# cam_gb = deprocess_image(cam_mask * gb)
# gb = deprocess_image(gb)

os.makedirs("../output_enhance_addi", exist_ok=True)

cam_output_path = os.path.join("../output_enhance_addi", f'{GradCAMPlusPlus}_cam.jpg')
# gb_output_path = os.path.join("../output_enhance_addi", f'{GradCAMPlusPlus}_gb.jpg')
# cam_gb_output_path = os.path.join("../output_enhance_addi", f'{GradCAMPlusPlus}_cam_gb.jpg')

cv2.imwrite(cam_output_path, cam_image)
# cv2.imwrite(gb_output_path, gb)
# cv2.imwrite(cam_gb_output_path, cam_gb)
