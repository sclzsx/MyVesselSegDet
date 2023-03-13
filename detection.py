import cv2
import torch
import numpy as np
from PIL import Image
# from skimage import io
import matplotlib.pyplot as plt
# from torchvision import datasets, transforms, 

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
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
        x = x.view(-1, 512*7*7)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x


input_size = 512
mean = [0.425753653049469, 0.29737451672554016, 0.21293757855892181]
std = [0.27670302987098694, 0.20240527391433716, 0.1686241775751114]
data_dir = 'datasets/eyepacs_binary_rgb/test'
transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std)])

# load the image
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# dataloader
dataloader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=1)

# model
model = VGG19(num_classes=1, pretrained=False)

checkpoint = torch.load('results/classification/binary_rgb/best_validation_weights.pt')
model.load_state_dict(checkpoint, strict=False)

# evaluation mode
model.eval()
model.cuda()
labels, preds = [], []
for img, label in tqdm(dataloader): 
    img = img.cuda()
    pred = model(img)
    
    # likelihood distribution
    distribution = pred

    pred = torch.where(pred > 0.5, 1, 0).squeeze(1)

    label = label.cpu().tolist()
    pred = pred.cpu().tolist()

    # print(label, pred)

    labels.extend(label)
    preds.extend(pred)

    print(distribution.shape)

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
    heatmap = torch.mean(activations, dim=1).squeeze()

    # relu to obtain only positive effect
    heatmap = torch.clamp(heatmap, min=0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    # show the heatmap
    with torch.no_grad():
        heatmap = np.array(heatmap.cpu())
    plt.matshow(heatmap)
    plt.show()

    # # read the input image
    # img = cv2.imread('./images/elephant/elephant.jpeg')
    # heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    # heatmap = np.uint8(255 * heatmap)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # superimposed_img = heatmap * 0.4 + img
    # cv2.imwrite('./heatmap_elephant.jpg', superimposed_img)

acc = accuracy_score(labels, preds)
p = precision_score(labels, preds)
r = recall_score(labels, preds)
f1 = f1_score(labels, preds)
conf = confusion_matrix(labels, preds)
metrics = {'accuracy': acc, 'precision': p, 'recall': r, 'f1_score': f1, 'confusion_matric': conf}
for metric in metrics.items():
    print(metric)


