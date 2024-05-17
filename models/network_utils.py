import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


class FeatureFusionNetwork(nn.Module):
    def __init__(self):
        super(FeatureFusionNetwork, self).__init__()
        self.fc1 = nn.Linear(512 + 512, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, img_features, pc_features):
        x = torch.cat((img_features, pc_features), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CustomLRScheduler:
    def __init__(self, optimizer, decay_steps, decay_rate):
        self.optimizer = optimizer
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.iteration = 0

    def step(self):
        self.iteration += 1
        if self.iteration % self.decay_steps == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.decay_rate


def evaluate_model(point_model, image_model, dataloader):
    point_model.eval()
    image_model.eval()

    cm_correct = 0
    cm_total = 0
    positive_distances = []
    negative_distances = []

    with torch.no_grad():
        # TODO: Change Here
        # ends up with 10 times test data size with half positive and half negative pairs
        for _ in range(3):
            for batch_id, (points, img1, img2, img3, y1, y2, y3) in enumerate(dataloader):
                points = points.transpose(2, 1)
                if torch.cuda.is_available():
                    points, img1, img2, img3 = points.cuda(), img1.cuda(), img2.cuda(), img3.cuda()
                    y1, y2, y3 = y1.cuda(), y2.cuda(), y3.cuda()

                fp, _ = point_model(points)
                fi1 = image_model(img1)  # anchor
                fi2 = image_model(img2)  # positive
                fi3 = image_model(img3)  # negative

                # Cross-Modality Correspondence (CM)
                dist1 = torch.norm(fp - fi1, dim=1)
                dist2 = torch.norm(fp - fi2, dim=1)
                dist3 = torch.norm(fp - fi3, dim=1)

                pred1 = (dist1 < dist3).float()
                pred2 = (dist2 < dist3).float()

                cm_correct += ((pred1 == y1).sum() + (pred2 == y2).sum()).item()
                cm_total += len(y1) + len(y2)

                # Cross-View Correspondence (CV)
                positive_distance = torch.norm(fi1 - fi2, dim=1)
                negative_distance = torch.norm(fi1 - fi3, dim=1)

                positive_distances.extend(positive_distance.cpu().numpy())
                negative_distances.extend(negative_distance.cpu().numpy())

    cm_accuracy = cm_correct / cm_total
    mpd_positive = np.mean(positive_distances)
    mpd_negative = np.mean(negative_distances)

    return cm_accuracy, mpd_positive, mpd_negative
