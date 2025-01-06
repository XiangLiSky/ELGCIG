import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
from PIL import Image
import os
import random
from sklearn.model_selection import ParameterGrid
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
root_dir = '../outputs_final_x/X/'

# 查找所有反事实数据集子目录
datasets = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

# 固定学习率
lr = 0.00001

# 超参数网格
param_grid = {
    'alpha': [i / 20 for i in range(21)]
}

results = []

# 遍历每个反事实数据集
for dataset_name in datasets:
    dataset_path = os.path.join(root_dir, dataset_name)
    counterfactual_dataset = ImageFolder(root=dataset_path, transform=transform)
    total_count = len(counterfactual_dataset)

    # 查找对应的验证集路径
    test_folder_path = f"../datasets/hardImageNet/val/{dataset_name}"

    # 加载验证集
    categories = []
    with open("../imagenet_classes.txt", "r") as f:
        for line in f:
            parts = line.strip().split(', ')
            if len(parts) == 2:
                categories.append(parts[1])
            else:
                print(f"Unexpected format in line: {line}")

    actual_class_index = categories.index(dataset_name.lower())

    # 定义不同的数据集大小（从5张开始，每次增加5张）
    data_sizes = list(range(20, total_count + 1, 20))

    # 遍历不同的数据集大小
    for size in data_sizes:
        print(f"Running experiment for {dataset_name} with dataset size: {size}")

        # 从完整数据集中抽取子集
        indices = list(range(total_count))
        random.shuffle(indices)
        subset_indices = indices[:size]
        subset_dataset = Subset(counterfactual_dataset, subset_indices)

        # 如果数据集小于一定的阈值，则不进行拆分，直接用于训练
        if size <= 10:
            train_dataset = subset_dataset
            val_dataset = subset_dataset
        else:
            train_count = int(0.8 * len(subset_dataset))
            val_count = len(subset_dataset) - train_count
            train_dataset, val_dataset = random_split(subset_dataset, [train_count, val_count])

        best_accuracy = 0.0
        best_params = None
        grid = ParameterGrid(param_grid)

        # 遍历所有 alpha 参数组合
        for params in grid:
            alpha = params['alpha']

            # 更新数据加载器
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

            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, 1000)
            model = model.to(device)

            # 替换最后一层（fc层）为新的全连接层，新层的参数默认是可训练的
            model.fc = torch.nn.Linear(num_ftrs, 1000)

            # 只有全连接层的参数是可训练的
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True

            model = model.to(device)

            # 定义损失函数和优化器
            criterion = torch.nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

            # 设定早停参数
            patience = 3
            min_delta = 0.001
            best_loss = float('inf')
            counter = 0
            num_epochs = 50
            model.eval()

            # 训练和验证过程
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
                print(f'Epoch {epoch + 1}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

                scheduler.step()

                if val_loss < best_loss - min_delta:
                    best_loss = val_loss
                    counter = 0
                    torch.save(model.state_dict(), '../output_enhance_x/best_model.pth')
                else:
                    counter += 1
                    if counter >= patience:
                        print(f'Early stopping after {epoch + 1} epochs.')
                        model.load_state_dict(torch.load('../output_enhance_x/best_model.pth'))
                        break

            # 加载最佳模型并进行FC层参数融合
            model.load_state_dict(torch.load('../output_enhance_x/best_model.pth'))

            with torch.no_grad():
                model.fc.weight.data = alpha * model.fc.weight.data + (1 - alpha) * original_fc_weight.to(device)
                model.fc.bias.data = alpha * model.fc.bias.data + (1 - alpha) * original_fc_bias.to(device)

            # 保存融合后的模型
            torch.save(model.state_dict(), '../output_enhance_x/resnet50_retrained_fused.pth')

            # 定义测试图像预处理
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
                        if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.Lg', '.jpeg')):
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
                    print(f"Trial #{trial + 1}: Accuracy for class '{dataset_name}': {accuracy * 100:.2f}%")

                average_accuracy = sum(accuracies) / trial_runs
                print(f"\nAverage Accuracy over {trial_runs} trials for {dataset_name}: {average_accuracy * 100:.2f}%")
                return average_accuracy

            # 计算当前超参数组合下的平均准确率
            current_accuracy = calculate_accuracy()
            results.append({'dataset': dataset_name, 'dataset_size': size, 'alpha': alpha, 'accuracy': current_accuracy})
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_params = params

results_df = pd.DataFrame(results)
results_df.to_csv('../output_enhance_x/enhance_results.csv', index=False)

# 计算每个类别的平均结果
average_results_df = results_df.groupby(['dataset_size', 'alpha']).mean().reset_index()
average_results_df.to_csv('../output_enhance_x/enhance_results.csv', index=False)

# 定义平滑函数
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# 绘制第一张图表：展示 alpha 对模型准确率的影响
plt.figure(figsize=(16, 9))
alpha_avg_df = average_results_df.groupby('alpha').mean().reset_index()
smoothed_accuracy = moving_average(alpha_avg_df['accuracy'].values)
plt.plot(alpha_avg_df['alpha'][len(alpha_avg_df['alpha']) - len(smoothed_accuracy):],
         smoothed_accuracy, label='Average Accuracy')

plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.title('Ablation Study: Alpha')
plt.legend()

# 保存图表为JPG文件
plt.savefig('../output_enhance_x/enhance_alpha_result.jpg')
plt.show()

# 绘制第二张图表：展示 Dataset Size 对模型准确率的影响
plt.figure(figsize=(16, 9))
dataset_size_avg_df = average_results_df.groupby('dataset_size').mean().reset_index()
smoothed_accuracy = moving_average(dataset_size_avg_df['accuracy'].values)
plt.plot(dataset_size_avg_df['dataset_size'][len(dataset_size_avg_df['dataset_size']) - len(smoothed_accuracy):],
         smoothed_accuracy, label='Average Accuracy')

plt.xlabel('Dataset Size')
plt.ylabel('Accuracy')
plt.title('Ablation Study: Dataset Size')
plt.legend()

# 保存图表为JPG文件
plt.savefig('../output_enhance_x/enhance_number_result.jpg')
plt.show()
