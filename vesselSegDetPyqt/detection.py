import cv2
import torch
import numpy as np
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
from pathlib import Path


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


def load_cls_model(pt_path):
    # model
    model = VGG19(num_classes=1, pretrained=False)

    checkpoint = torch.load(pt_path)
    model.load_state_dict(checkpoint, strict=False)

    # evaluation mode
    model.eval()
    # model.cuda()
    model.cpu()
    return model


def pre_process_for_cls(image_path, input_size, mean, std):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std)])

    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0)

    label = int(Path(image_path).name.split('_')[0])

    img_np = np.array(img_tensor.detach().cpu().squeeze().permute(1, 2, 0))
    img_np = (np.clip(img_np * std + mean, 0, 1) * 255).astype('uint8')
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    return img_tensor, img_np, label


def do_classify(img_tensor, model, classify_threshold):
    # img_tensor = img_tensor.cuda()
    img_tensor = img_tensor.cpu()
    pred = model(img_tensor)

    # likelihood distribution
    distribution = pred

    pred_score = pred.cpu().squeeze().item()

    pred = 1 if pred_score > classify_threshold else 0

    # gradient of the output with respect to the model parameters
    distribution[:, 0].backward()

    # gradients that we hooked (gradients of the last conv)
    gradients = model.get_activation_gradient()
    # print(gradients.shape)

    # pool the gradients across the channel
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    # print(len(pooled_gradients))

    # activations of the last conv layer
    activations = model.get_activation(img_tensor).detach()

    # weight the channels by corresponding gradients
    for i in range(len(pooled_gradients)):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average all channels of the weighted activations
    heatmap = torch.mean(activations, dim=1).squeeze().cpu()

    # relu to obtain only positive effect
    heatmap = torch.clamp(heatmap, min=0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    heatmap = np.array(heatmap)
    heatmap = (cv2.resize(heatmap, (img_tensor.shape[-2], img_tensor.shape[-1])) * 255).astype('uint8')

    return heatmap, pred


def do_detect(img_np, heatmap, pred, anomaly_threshold):
    det_np = img_np
    max_bbox = [0, 0, 0, 0]
    if pred == 1:
        anomaly_np = np.where(heatmap > anomaly_threshold, 255, 0).astype('uint8')
        _, contours, _ = cv2.findContours(anomaly_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            max_area = -1
            max_contour = None
            for c in contours:
                area = cv2.contourArea(c)
                if area > max_area:
                    max_contour = c
                    max_area = area
            [x, y, w, h] = cv2.boundingRect(max_contour)
            max_bbox = [x, y, w, h]
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            cv2.rectangle(det_np, pt1, pt2, (0, 255, 0), 2)
    return det_np, max_bbox


if __name__ == '__main__':
    pt_path = 'weights/classification.pt'
    input_size = 512
    mean = [0.425753653049469, 0.29737451672554016, 0.21293757855892181]
    std = [0.27670302987098694, 0.20240527391433716, 0.1686241775751114]
    classify_threshold = 0.5
    anomaly_threshold = 0.5

    for image_path in Path('images').glob('*.png'):
        image_path = str(image_path)

        model = load_cls_model(pt_path)

        img_tensor, img_np, label = pre_process_for_cls(image_path, input_size, mean, std)

        heatmap, pred = do_classify(img_tensor, model, classify_threshold)

        det_np, max_bbox = do_detect(img_np, heatmap, pred, anomaly_threshold)

        show_info = 'label is: {}, pred is: {}, bbox is:{}'.format(label, pred, max_bbox)

        print(show_info)
        cv2.imshow('det_np', det_np)
        cv2.waitKey()
