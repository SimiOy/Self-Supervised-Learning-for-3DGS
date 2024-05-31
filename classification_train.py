import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import SupervisedTrainingModelNetDataLoader
from data_loader.ModelNetDataLoader import Config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# point_dir = 'C:/Gaussian-Splatting/gaussian-splatting/output/modelNet10/'
# img_dir = 'C:/ResearchProject/datasets/modelnet10/ModelNet10_captures'
# log_dir = 'pointnet_uniform_sampling'
# batch_size = 32
# num_workers = 8
# num_views = 64

classes = {
    'bathtub': 0,
    'bed': 1,
    'chair': 2,
    'desk': 3,
    'dresser': 4,
    'monitor': 5,
    'night_stand': 6,
    'sofa': 7,
    'table': 8,
    'toilet': 9
}


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    # Architecture
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')

    # Model params
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--num_workers', type=int, default=8, help='number of data loader workers')
    parser.add_argument('--num_views', type=int, default=1, help='number of distinct views')

    # Data Loader Args
    parser.add_argument('--point_dir', type=str, required=True, help='3D Gaussian Splats Dir')
    parser.add_argument('--img_dir', type=str, required=True, help='Renders/Images Dir')

    return parser.parse_args()


def read_config(log_dir):
    log_path = Path(ROOT_DIR, 'log', log_dir, 'logs')
    for file in log_path.iterdir():
        if file.suffix == '.txt':
            with open(file, 'r') as f:
                lines = f.readlines()
                # Assuming the configuration is on the second line
                config_line = lines[1]
                # Extract the part after 'INFO -' which contains the Namespace string
                namespace_str = config_line.split('- INFO - ')[1].strip()
                # Converting Namespace string to a dictionary
                namespace_str = namespace_str.replace('Namespace(', '').replace(')', '')
                config_dict = dict(item.split('=') for item in namespace_str.split(', '))
                # Post-process values to handle Boolean and None values
                for key, value in config_dict.items():
                    if value.isdigit():
                        config_dict[key] = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        config_dict[key] = float(value)
                    elif value == 'True':
                        config_dict[key] = True
                    elif value == 'False':
                        config_dict[key] = False
                    elif value == 'None':
                        config_dict[key] = None
                    else:
                        # Stripping extra quotes from string literals
                        config_dict[key] = value.strip("'").strip('"')
                return config_dict
    raise FileNotFoundError("No configuration file found in the specified directory.")


def extract_features(data_loader, pointnet, resnet18):
    point_features, image_features, labels = [], [], []
    with torch.no_grad():
        for batch_id, (points, imgs, cls_name) in tqdm(enumerate(data_loader, 0), total=len(data_loader)):
            points = points.transpose(2, 1).cuda()
            fp, _ = pointnet(points)
            point_features.append(fp.cpu().numpy())

            # Max pooling across the view features
            # Have to move to CPU and to NumPy for SVM ScikitLearn to work
            img_features = [resnet18(img.cuda()).cpu().numpy() for img in imgs]
            max_pooled_features = np.max(np.stack(img_features), axis=0)
            image_features.append(max_pooled_features)

            labels.extend(cls_name)

    point_features = np.vstack(point_features)
    image_features = np.vstack(image_features)
    labels = np.array([classes[label] for label in labels])
    return point_features, image_features, labels


def train_svm(features, labels):
    svm = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1))
    # svm = SVC(kernel='linear', C=1)
    svm.fit(features, labels)
    return svm


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = read_config(args.log_dir)

    '''ACCESS LOG DIR'''
    exp_dir = Path('./log/')
    exp_dir = exp_dir.joinpath(args.log_dir)

    '''DATA LOADING'''
    data_loader_config = Config(config['num_point'], config['use_normals'], config['use_scale_and_rotation'],
                                config['use_colors'], config['furthest_point_sample'])
    train_dataset = SupervisedTrainingModelNetDataLoader(point_dir=args.point_dir, img_dir=args.img_dir,
                                                         args=data_loader_config, split='train',
                                                         num_views=args.num_views)
    test_dataset = SupervisedTrainingModelNetDataLoader(point_dir=args.point_dir, img_dir=args.img_dir,
                                                        args=data_loader_config, split='test', num_views=args.num_views)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.num_workers, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers)

    '''MODEL LOADING'''
    pointnet_model = importlib.import_module(config['model'])

    no_features = 3
    if config['use_normals']:
        no_features += 3
    if config['use_colors']:
        no_features += 44
    if config['use_scale_and_rotation']:
        no_features += 7

    pointnet = pointnet_model.get_model(no_features=no_features, normal_channel=config['use_normals'])
    resnet18 = models.resnet18()
    resnet18.fc = nn.Identity()  # Remove the last fully connected layer

    pointnet = pointnet.cuda()
    resnet18 = resnet18.cuda()

    checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
    pointnet.load_state_dict(checkpoint['pointnet_state_dict'])
    resnet18.load_state_dict(checkpoint['resnet18_state_dict'])

    pointnet.eval()
    resnet18.eval()

    # Feature extraction for training data
    train_point_features, train_image_features, train_labels = extract_features(trainDataLoader, pointnet, resnet18)

    # Train SVMs
    svm_point = train_svm(train_point_features, train_labels)
    svm_image = train_svm(train_image_features, train_labels)
    print("SVMs trained.")

    # Feature extraction for test data
    test_point_features, test_image_features, test_labels = extract_features(testDataLoader, pointnet, resnet18)

    # Evaluating SVMs
    point_predictions = svm_point.predict(test_point_features)
    image_predictions = svm_image.predict(test_image_features)

    print("Point-based SVM Accuracy:", accuracy_score(test_labels, point_predictions))
    print("Image-based SVM Accuracy:", accuracy_score(test_labels, image_predictions))
