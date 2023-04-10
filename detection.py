import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from torchvision import models, transforms, datasets
import torch.nn as nn


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

        self.gradients = None

    # hook
    def activation_hook(self, grad):
        self.gradients = grad

    # extract gradient
    def get_activation_gradient(self):
        return self.gradients

    # extract the activation after the last ReLU
    def get_activation(self, x):
        return self.features(x)

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)

        # register the hook in the forward pass
        hook = x.register_hook(self.activation_hook)

        x = self.max_pool(x)
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(-1, 512 * 7 * 7)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x


def detection(data_dir, pt_path, input_size, mean, std, save_dir, anomaly_threshold, pred_num):
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

    for cnt, (img, label) in enumerate(dataloader):
        if cnt > pred_num and pred_num > 0:
            break

        img = img.cuda()
        pred = model(img)

        # likelihood distribution
        distribution = pred

        label = label.cpu().squeeze().item()
        pred_score = pred.cpu().squeeze().item()

        pred = 1 if pred_score > 0.5 else 0

        labels.append(label)
        preds.append(pred)

        # print(distribution.shape)

        # gradient of the output with respect to the model parameters
        distribution[:, 0].backward()

        # gradients that we hooked (gradients of the last conv)
        gradients = model.get_activation_gradient()
        # print(gradients.shape)

        # pool the gradients across the channel
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        # print(len(pooled_gradients))

        # activations of the last conv layer
        activations = model.get_activation(img).detach()

        # weight the channels by corresponding gradients
        for i in range(len(pooled_gradients)):
            activations[:, i, :, :] *= pooled_gradients[i]

        # average all channels of the weighted activations
        heatmap = torch.mean(activations, dim=1).squeeze().cpu()

        # relu to obtain only positive effect
        heatmap = torch.clamp(heatmap, min=0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        img_np = np.array(img.detach().cpu().squeeze().permute(1, 2, 0))
        img_np = (np.clip(img_np * std + mean, 0, 1) * 255).astype('uint8')
        det = img_np

        H, W, C = img_np.shape

        fig = plt.figure(figsize=(24, 8))

        ax = fig.add_subplot(131)
        ax.set_title('Image')
        ax.axis('off')
        ax.imshow(img_np)

        heatmap = np.array(heatmap)
        heatmap = (cv2.resize(heatmap, (W, H)) * 255).astype('uint8')

        ax = fig.add_subplot(132)
        ax.set_title('Heatmap')
        ax.axis('off')
        ax.imshow(heatmap)

        ax = fig.add_subplot(133)
        ax.axis('off')
        ax.set_title('Label:{}    Prediction:{}'.format(label, pred))
        if pred == 1:
            anomaly_np = np.where(heatmap > anomaly_threshold, 255, 0).astype('uint8')
            _, contours, _ = cv2.findContours(anomaly_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                max_area = -1
                for c in contours:
                    area = cv2.contourArea(c)
                    if area > max_area:
                        max_contour = c
                        max_area = area
                    [x, y, w, h] = cv2.boundingRect(max_contour)
                rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
        ax.imshow(img_np)
        plt.savefig(save_dir + '/' + str(cnt) + '.jpg')
        # plt.show()
        plt.close()

    acc = accuracy_score(labels, preds)
    p = precision_score(labels, preds)
    r = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    conf = confusion_matrix(labels, preds)
    metrics = {'accuracy': acc, 'precision': p, 'recall': r, 'f1_score': f1, 'confusion_matric': conf}
    for metric in metrics.items():
        print(metric)


if __name__ == '__main__':
    pt_path = 'results/classification/binary_rgb/best_validation_weights.pt'
    input_size = 512
    mean = [0.425753653049469, 0.29737451672554016, 0.21293757855892181]
    std = [0.27670302987098694, 0.20240527391433716, 0.1686241775751114]
    data_dir = 'datasets/eyepacs_binary_rgb/test'
    pred_num = 100
    save_dir = 'results/detection'
    anomaly_threshold = 0.5

    detection(data_dir, pt_path, input_size, mean, std, save_dir, anomaly_threshold, pred_num)
