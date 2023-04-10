import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from torchvision import models, transforms, datasets
import torch.nn as nn
import json
from tqdm import tqdm
# 导包
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class VGG19(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(VGG19, self).__init__()

        self.features = models.vgg19(pretrained=pretrained).features[:36]

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.max_pool(x)
        x = self.avgpool(x)
        x = x.view(-1, 512 * 7 * 7)
        x = self.classifier(x)
        return x


def detection(data_dir, pt_path, input_size, mean, std, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std)])

    # load the image
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=1)

    # model
    model = VGG19(num_classes=1, pretrained=False)

    checkpoint = torch.load(pt_path)
    model.load_state_dict(checkpoint, strict=False)

    # evaluation mode
    model.eval()
    model.cuda()
    labels, preds = [], []
    scores = []
    for img, label in tqdm(dataloader):
        img = img.cuda()
        with torch.no_grad():
            pred = model(img)

        label = label.cpu().squeeze().item()
        pred_score = pred.cpu().squeeze().item()
        # print(pred_score)
        scores.append(pred_score)
        pred = 1 if pred_score > 0.5 else 0

        labels.append(label)
        preds.append(pred)

    fpr, tpr, thread = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_dir + '/roc.png', )
    plt.show()

    acc = accuracy_score(labels, preds)
    p = precision_score(labels, preds)
    r = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    metrics = {'accuracy': acc, 'precision': p, 'recall': r, 'f1_score': f1}
    for metric in metrics.items():
        print(metric)
    with open(save_dir + '/metric.json', 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    input_size = 512
    mean = [0.425753653049469, 0.29737451672554016, 0.21293757855892181]
    std = [0.27670302987098694, 0.20240527391433716, 0.1686241775751114]

    pt_path = 'results/classification/binary_rgb/best_validation_weights.pt'
    data_dir = 'datasets/eyepacs_binary_rgb/test'
    save_dir = 'results/detectionMetric/rgb'
    detection(data_dir, pt_path, input_size, mean, std, save_dir)

    pt_path = 'results/classification/binary_laddernet_sharpen/best_validation_weights.pt'
    data_dir = 'datasets/eyepacs_binary_laddernet_sharpen/test'
    save_dir = 'results/detectionMetric/sharpen'
    detection(data_dir, pt_path, input_size, mean, std, save_dir)