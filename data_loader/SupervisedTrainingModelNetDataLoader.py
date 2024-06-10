import os
import re

import numpy as np
import open3d as o3d
import torch
from PIL import Image
from pyntcloud import PyntCloud
from torch.utils.data import Dataset
from torchvision import transforms

from data_loader.ModelNetDataLoader import Config
from data_loader.dataloader_utils import farthest_point_sample, pc_normalize


class SupervisedTrainingModelNetDataLoader(Dataset):
    def __init__(self, point_dir, img_dir, args, split='train', opacity_threshold=0.3, num_views=1, class_fraction=1,
                 novel_views=False):
        self.num_points = args.num_point
        self.fps = args.furthest_point_sample
        self.use_normals = args.use_normals
        self.use_colors = args.use_colors
        self.use_sr = args.use_scale_and_rotation
        self.opacity_threshold = opacity_threshold
        self.num_views = num_views
        self.class_fraction = class_fraction
        self.novel_views = novel_views

        assert split in ['train', 'test']
        np.random.seed(42)  # to get always the same split

        class_paths = [os.path.join(point_dir, class_name) for class_name in os.listdir(point_dir)]
        self.model_paths = []
        self.img_paths = {}

        # fore each class label
        for class_path in class_paths:
            class_name = os.path.basename(class_path)
            models_in_class = os.listdir(os.path.join(class_path, split))

            # compute fraction of models to select
            num_models_in_class = len(models_in_class)
            num_models_to_select = int(self.class_fraction * num_models_in_class)
            selected_models = np.random.choice(models_in_class, size=num_models_to_select, replace=False)

            # add only these
            for model_name in selected_models:
                model_path = os.path.join(class_path, split, model_name)
                self.model_paths.append((model_path, model_name))

                # grab the images from the point_dir reconstruction folder
                if split == 'test' and novel_views:
                    self.img_paths[model_name] = (os.path.join(model_path, 'train', 'ours_15000'), class_name)
                else:
                    self.img_paths[model_name] = (os.path.join(img_dir, class_name, split, model_name, 'images'), class_name)

        self.transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.model_paths)

    def __getitem__(self, idx):
        # Load 3DGS model
        model_path, model_name = self.model_paths[idx]
        point_cloud_path = self._get_latest_point_cloud_path(model_path)
        point_cloud: PyntCloud = PyntCloud.from_file(point_cloud_path)

        # Extract point properties and convert them to tensors
        points = point_cloud.points[['x', 'y', 'z']].values.astype(np.float32)
        opacity = point_cloud.points['opacity'].values.astype(np.float32)

        if self.use_normals:
            normals = point_cloud.points[['nx', 'ny', 'nz']].values.astype(np.float32)
            points = np.concatenate((points, normals), axis=1)

        # Assuming spherical harmonics coefficients are stored as 'f_rest_i'
        if self.use_colors:
            sh_keys = [f'f_rest_{i}' for i in range(44)]
            spherical_harmonics = point_cloud.points[sh_keys].values.astype(np.float32)
            points = np.concatenate((points, spherical_harmonics), axis=1)

        # Other properties
        if self.use_sr:
            scale = point_cloud.points[['scale_0', 'scale_1', 'scale_2']].values.astype(np.float32)
            rotation = point_cloud.points[['rot_0', 'rot_1', 'rot_2', 'rot_3']].values.astype(np.float32)
            points = np.concatenate((points, scale, rotation), axis=1)

        # perform dynamic filtering and sub-sampling
        mask = opacity > self.opacity_threshold
        properties = points[mask, :]

        if self.fps:
            sampled_properties = farthest_point_sample(properties, self.num_points)
        else:
            indices = np.random.choice(len(properties), self.num_points)
            sampled_properties = properties[indices]

        # normalize position of points
        sampled_properties[:, 0:3] = torch.tensor(pc_normalize(sampled_properties[:, 0:3]))

        # Load images
        img_dir, cls_name = self.img_paths[model_name]
        img_indices = np.random.choice(64, self.num_views, replace=False)
        if self.novel_views:
            # padded with 5 zeros by default
            imgs = [self.transform(Image.open(os.path.join(img_dir, f'{idx:05}.png')).convert('RGB')) for idx in
                    img_indices]
        else:
            imgs = [self.transform(Image.open(os.path.join(img_dir, f'{idx:03}.png')).convert('RGB')) for idx in
                    img_indices]

        return sampled_properties, imgs, cls_name

    def _get_latest_point_cloud_path(self, model_path):
        """Finds the path of the latest point cloud by iteration number."""
        point_cloud_dir = os.path.join(model_path, 'point_cloud')
        iteration_dirs = [name for name in os.listdir(point_cloud_dir) if
                          os.path.isdir(os.path.join(point_cloud_dir, name))]

        # Extract the numeric part of the directory names using regex
        def extract_iteration_number(name):
            match = re.search(r'\d+', name)
            return int(match.group()) if match else -1

        # Find the directory with the highest numeric value
        latest_iteration_name = max(iteration_dirs, key=extract_iteration_number)

        # Return the path to the point cloud file inside the chosen iteration folder
        latest_iteration_dir = os.path.join(point_cloud_dir, latest_iteration_name)
        point_cloud_path = os.path.join(latest_iteration_dir, 'point_cloud.ply')

        return point_cloud_path

    def visualize_point_cloud(self, index):
        """ Visualizes the point cloud at the given index """
        point_cloud, _, _, _, _, _, _, label = self.__getitem__(index)
        print(f'Doing model {label}')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, 0:3])
        o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    config = Config(num_point=4096, furthest_point_sample=True, use_normals=False)
    # Usage example:
    point_dir = 'C:/Gaussian-Splatting/gaussian-splatting/output/modelNet10/'
    img_dir = 'C:/ResearchProject/datasets/modelnet10/ModelNet10_captures'
    dataset = SupervisedTrainingModelNetDataLoader(point_dir, img_dir, config)
    dataset.visualize_point_cloud(515)
