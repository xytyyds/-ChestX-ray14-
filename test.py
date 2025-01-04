import torch
import numpy as np
import csv
from dataset import CheXDataset, data_generate
from torch.utils import data
from net.model import DenseNet121_torch_version
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt


def calculate_roc_auc(true_labels, predictions):
    roc_auc_scores = []
    for i in range(true_labels.shape[1]):
        if len(np.unique(true_labels[:, i])) == 1:
            print(f"Skipping ROC AUC calculation for class {i}: only one class present in y_true")
            continue
        try:
            roc_auc = roc_auc_score(true_labels[:, i], predictions[:, i])
            roc_auc_scores.append(roc_auc)
        except ValueError as e:
            print(f"Error calculating ROC AUC for class {i}: {e}")
            continue
    return np.mean(roc_auc_scores) if roc_auc_scores else None


def test_model(model, test_loader, device):
    model.eval()
    test_loss = 0
    true_labels = []
    predictions = []

    model_loss = torch.nn.BCELoss(reduction='mean')

    print("Starting evaluation on test data...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            print(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = model_loss(outputs, labels)

            test_loss += loss.item()
            true_labels.append(labels.cpu().numpy())
            predictions.append(outputs.cpu().numpy())

            if np.isnan(test_loss):
                print("Encountered NaN value in test loss. Stopping evaluation.")
                break

    test_loss /= len(test_loader)
    true_labels = np.concatenate(true_labels)
    predictions = np.concatenate(predictions)

    print("True labels shape:", true_labels.shape)
    print("Predictions shape:", predictions.shape)

    accuracy = accuracy_score(true_labels, (predictions > 0.5).astype(int))
    roc_auc = calculate_roc_auc(true_labels, predictions)

    print(f"Test Loss for this evaluation: {test_loss}")
    print(f"Test Accuracy for this evaluation: {accuracy}")
    if roc_auc is not None:
        print(f"Test ROC AUC for this evaluation: {roc_auc}")
    else:
        print("ROC AUC could not be calculated for this evaluation.")

    return test_loss, accuracy, roc_auc


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device used: {device}')

    print("Loading CSV file...")
    readFile = open("train.csv", "r", newline='')
    reader = csv.reader(readFile)

    print("Generating test data...")
    _, _, test_images, test_labels = data_generate(reader)
    test_images = test_images
    test_labels = test_labels
    print(f"Number of test samples: {len(test_images)}")

    testDataset = CheXDataset(test_images, test_labels)
    test_loader = data.DataLoader(testDataset, batch_size=32, shuffle=False)

    print("Loading model...")
    model_path = './weights/CheXNet.pth'
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.to(device)

    test_losses = []
    accuracies = []
    # 确保记录多个数据点
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Starting test round {epoch + 1}")
        test_loss, accuracy, _ = test_model(model, test_loader, device)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

    print(f"Final Test Loss: {test_losses[-1]:.3f}, Final Test Accuracy: {accuracies[-1]:.3f}")

    # 打印出要绘制的数据点，进行调试
    print("Test Losses:", test_losses)
    print("Test Accuracies:", accuracies)

    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), test_losses, label='Test Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Testing Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), accuracies, label='Test Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Testing Accuracy')
    plt.legend()

    plt.show()
