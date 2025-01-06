import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from PIL import Image
import os
import random

# 设置随机种子
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 根目录路径
root_dir = '../additional_exp/Cam'
test_folder_path = "../additional_exp/val/baseball"

categories = []
with open("../imagenet_classes.txt", "r") as f:
    for line in f:
        parts = line.strip().split(', ')
        if len(parts) == 2:
            categories.append(parts[1])
        else:
            print(f"Unexpected format in line: {line}")
actual_class_index = categories.index("baseball")


# 固定超参数
lr = 0.00001
alpha = 0.9

# 结果记录
results = []

# 加载反事实数据集子目录
datasets = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

# 遍历每个反事实数据集
for dataset_name in datasets:
    dataset_path = os.path.join(root_dir, dataset_name)
    dataset = ImageFolder(root=dataset_path, transform=transform)
    total_count = len(dataset)
    indices = list(range(total_count))
    random.shuffle(indices)
    subset_indices = indices[:20]
    subset_dataset = Subset(dataset, subset_indices)

    # 划分训练集和测试集
    train_size = int(0.8 * len(subset_dataset))
    val_size = len(subset_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(subset_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=9, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=9, shuffle=False)

    # 加载预训练的ResNet50模型
    model = models.resnet50(pretrained=True)

    # 保存原始全连接层的参数
    original_fc_weight = model.fc.weight.clone().detach()
    original_fc_bias = model.fc.bias.clone().detach()

    # 冻结所有层的参数
    for param in model.parameters():
        param.requires_grad = False

    # 替换最后一层为新的全连接层
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 1000)
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 设置早停机制
    patience = 3
    min_delta = 0.001
    best_loss = float('inf')
    counter = 0
    num_epochs = 50
    model.eval()
    # 训练过程
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_images)
                v_loss = criterion(val_outputs, val_labels)
                val_loss += v_loss.item() * val_images.size(0)

                _, predicted = torch.max(val_outputs, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100 * correct / total
        print(
            f'Epoch {epoch + 1}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        scheduler.step()

        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), '../output_enhance_addi/best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping after {epoch + 1} epochs.')
                model.load_state_dict(torch.load('../output_enhance_addi/best_model.pth'))
                break

        # 加载最佳模型并进行FC层参数融合
    model.load_state_dict(torch.load('../output_enhance_addi/best_model.pth'))

    with torch.no_grad():
        model.fc.weight.data = alpha * model.fc.weight.data + (1 - alpha) * original_fc_weight.to(device)
        model.fc.bias.data = alpha * model.fc.bias.data + (1 - alpha) * original_fc_bias.to(device)

    # 保存融合后的模型
    model_path = f'../output_enhance_addi/{dataset_name}_retrained_fused.pth'
    torch.save(model.state_dict(), model_path)

    test_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def calculate_accuracy(trial_runs=10):
        accuracies = []
        image_paths = os.listdir(test_folder_path)
        image_count = 47

        for trial in range(trial_runs):
            selected_images = random.sample(image_paths, image_count)

            correct_predictions = 0
            total_predictions = 0

            for image_name in selected_images:
                image_path = os.path.join(test_folder_path, image_name)
                if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.Lg', '.jpeg', 'JPEG')):
                    image = Image.open(image_path)
                    input_tensor = test_preprocess(image)
                    input_batch = input_tensor.unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(input_batch)

                    _, predicted_idx = torch.max(output, 1)
                    predicted_idx = predicted_idx.item()

                    total_predictions += 1
                    if predicted_idx == actual_class_index:
                        correct_predictions += 1

            accuracy = correct_predictions / total_predictions
            accuracies.append(accuracy)
            print(f"Trial #{trial + 1}: Accuracy for class baseball: {accuracy * 100:.2f}%")

        average_accuracy = sum(accuracies) / trial_runs
        print(f"\nAverage Accuracy over {trial_runs} trials for {dataset_name}: {average_accuracy * 100:.2f}%")
        return average_accuracy


    accuracy = calculate_accuracy()
    # 保存准确率结果到文件
    with open('../output_enhance_addi/results.txt', 'a') as f:
        f.write(f"Dataset: {dataset_name}, Accuracy: {accuracy * 100:.2f}%, Model Path: {model_path}\n")

print("All datasets processed. Results saved to ../output_enhance_addi/results.txt")
