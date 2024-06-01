import os
import re

import numpy as np
import open3d as o3d
import torch
from PIL import Image
from pyntcloud import PyntCloud
from torch.utils.data import Dataset
from torchvision import transforms

from data_loader.dataloader_utils import farthest_point_sample, pc_normalize


class SupervisedTrainingModelNetDataLoader(Dataset):
    def __init__(self, point_dir, img_dir, args, split='train', opacity_threshold=0.3, num_views=1):
        self.num_points = args.num_point
        self.fps = args.furthest_point_sample
        self.use_normals = args.use_normals
        self.use_colors = args.use_colors
        self.use_sr = args.use_scale_and_rotation
        self.opacity_threshold = opacity_threshold
        self.num_views = num_views

        # TODO: Use the model_path from the cfg_args of each model instead
        assert split in ['train', 'test']

        class_paths = [os.path.join(point_dir, class_name) for class_name in os.listdir(point_dir)]
        self.model_paths = [(os.path.join(cls_path, split, model_name), model_name)
                            for cls_path in class_paths
                            for model_name in os.listdir(os.path.join(cls_path, split))]

        self.img_paths = {model_name: (os.path.join(img_dir, class_name, split, model_name, 'images'), class_name)
                          for class_name in os.listdir(img_dir)
                          for model_name in os.listdir(os.path.join(img_dir, class_name, split))}

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
