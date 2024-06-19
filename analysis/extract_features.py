import os
import re

import pandas as pd
from pyntcloud import PyntCloud


def _get_latest_point_cloud_path(model_path):
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


def retrieve_all_features(save_dir, model_paths):
    all_features_df = pd.DataFrame()
    for idx, (model_path, model_name, label) in enumerate(model_paths):
        point_cloud_path = _get_latest_point_cloud_path(model_path)
        point_cloud = PyntCloud.from_file(point_cloud_path)

        features_df = pd.DataFrame()

        # features_df['x'], features_df['y'], features_df['z'] = point_cloud.points[['x', 'y', 'z']].values.T
        # features_df[['nx', 'ny', 'nz']] = point_cloud.points[['nx', 'ny', 'nz']].values
        features_df['opacity'] = point_cloud.points['opacity'].values
        # features_df[['scale_0', 'scale_1', 'scale_2']] = point_cloud.points[['scale_0', 'scale_1', 'scale_2']].values
        # features_df[['rot_0', 'rot_1', 'rot_2', 'rot_3']] = point_cloud.points[
        #     ['rot_0', 'rot_1', 'rot_2', 'rot_3']].values

        # for i in range(44):
        #     features_df[f'f_rest_{i}'] = point_cloud.points[f'f_rest_{i}'].values

        features_df['name'] = model_name
        features_df['label'] = label
        all_features_df = pd.concat([all_features_df, features_df], ignore_index=True)

    all_features_df.to_csv(save_dir, index=False)
    print("Features saved to:", save_dir)


def load_features_from_csv():
    df = pd.read_csv('model_features.csv')
    return df


root_dir = 'C:/Gaussian-Splatting/gaussian-splatting/output/modelNet10/'
save_dir = 'C:/ResearchProject/CV3dgs/model_features.csv'

if __name__ == '__main__':
    class_paths = [(os.path.join(root_dir, name), name) for name in os.listdir(root_dir)]
    model_paths = []
    for cls_path, label in class_paths:
        for split in ['train', 'test']:
            split_path = os.path.join(cls_path, split)
            for model_name in os.listdir(split_path):
                full_path = os.path.join(split_path, model_name)
                model_paths.append((full_path, model_name, label))

    retrieve_all_features(save_dir, model_paths)
    features_df = load_features_from_csv()
    print("Loaded features DataFrame for analysis:")
    print(features_df.head())
