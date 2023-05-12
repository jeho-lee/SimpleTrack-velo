# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
import mmcv
import os
import argparse
from nuscenes.nuscenes import NuScenes
from PIL import Image
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from typing import Tuple, List, Iterable
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.detection.data_classes import DetectionBox
# from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.common.utils import boxes_to_sensor
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.render import visualize_sample









import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
import copy
from matplotlib.axes import Axes
from typing import Tuple, List, Iterable

def map_point_cloud_to_image(pc, image, ego_pose, cam_pose, cam_intrinsics, min_dist=1.0):
    """ map a global coordinate point cloud to image
    Args:
        pc (numpy.ndarray [N * 3])
    """
    point_cloud = copy.deepcopy(pc)
    
    # transform point cloud to the ego
    point_cloud -= ego_pose[:3, 3]
    point_cloud = point_cloud @ ego_pose[:3, :3]

    # transform from ego to camera
    point_cloud -= cam_pose[:3, 3]
    point_cloud = point_cloud @ cam_pose[:3, :3]

    # project points to images
    # step 1. Depth and colors
    depths = point_cloud[:, 2]
    intensities = point_cloud[:, 2]
    intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
    intensities = intensities ** 0.1
    intensities = np.maximum(0, intensities - 0.5)
    coloring = intensities

    # step 2. Project onto images with intrinsics
    points = point_cloud.T
    points = view_points(points[:3, :], cam_intrinsics, normalize=True).T

    # step 3. Remove the points that are outside/behind the camera
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[:, 0] > 1)
    mask = np.logical_and(mask, points[:, 0] < image.size[0] - 1)
    mask = np.logical_and(mask, points[:, 1] > 1)
    mask = np.logical_and(mask, points[:, 1] < image.size[1] - 1)
    points = points[mask, :]
    coloring = coloring[mask]

    return points, coloring

class NuscenesTrackingBox(Box):
    """ Data class used during tracking evaluation. Can be a prediction or ground truth."""

    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_translation: Tuple[float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
                 num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 tracking_id: str = '',  # Instance id of this object.
                 tracking_name: str = '',  # The class name used in the tracking challenge.
                 tracking_score: float = -1.0):  # Does not apply to GT.

        super().__init__(translation, size, rotation, np.nan, tracking_score, name=tracking_id)

        assert tracking_name is not None, 'Error: tracking_name cannot be empty!'

        assert type(tracking_score) == float, 'Error: tracking_score must be a float!'
        assert not np.any(np.isnan(tracking_score)), 'Error: tracking_score may not be NaN!'

        # Assign.
        self.tracking_id = tracking_id
        self.tracking_name = tracking_name
        self.tracking_score = tracking_score

    def __eq__(self, other):
        return (self.sample_token == other.sample_token and
                self.translation == other.translation and
                self.size == other.size and
                self.rotation == other.rotation and
                self.velocity == other.velocity and
                self.ego_translation == other.ego_translation and
                self.num_pts == other.num_pts and
                self.tracking_id == other.tracking_id and
                self.tracking_name == other.tracking_name and
                self.tracking_score == other.tracking_score)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'tracking_id': self.tracking_id,
            'tracking_name': self.tracking_name,
            'tracking_score': self.tracking_score
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']),
                   ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                   else tuple(content['ego_translation']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                   tracking_id=content['tracking_id'],
                   tracking_name=content['tracking_name'],
                   tracking_score=-1.0 if 'tracking_score' not in content else float(content['tracking_score']))
    
    def render(self,
               axis: Axes,
               view: np.ndarray = np.eye(3),
               normalize: bool = False,
               colors: Tuple = ('b', 'r', 'k'),
               linestyle: str = 'solid',
               linewidth: float = 2,
               text=True) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth, linestyle=linestyle)
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=colors, linewidth=linewidth, linestyle=linestyle)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors)
        draw_rect(corners.T[4:], colors)

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=colors, linewidth=linewidth, linestyle=linestyle)
        corner_index = np.random.randint(0, 8, 1)
        if text:
            axis.text(corners[0, corner_index] - 1, corners[1, corner_index] - 1, self.tracking_id, color=colors, fontsize=8)
            
            
            
            
            
            
            
            

COLOR_MAP = {
    'red': np.array([191, 4, 54]) / 256,
    'light_blue': np.array([4, 157, 217]) / 256,
    'black': np.array([0, 0, 0]) / 256,
    'gray': np.array([140, 140, 136]) / 256,
    'purple': np.array([224, 133, 250]) / 256, 
    'dark_green': np.array([32, 64, 40]) / 256,
    'green': np.array([77, 115, 67]) / 256,
    'brown': np.array([164, 103, 80]) / 256,
    'light_green': np.array([135, 206, 191]) / 256,
    'orange': np.array([229, 116, 57]) / 256,
}
COLOR_KEYS = list(COLOR_MAP.keys())


cams = ['CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_FRONT_LEFT']

import numpy as np
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from PIL import Image
from matplotlib import rcParams


def get_predicted_data(sample_data_token: str,
                       box_vis_level: BoxVisibility = BoxVisibility.ANY,
                       selected_anntokens=None,
                       use_flat_vehicle_coordinates: bool = False,
                       pred_anns=None
                       ):
    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    # if selected_anntokens is not None:
    #    boxes = list(map(nusc.get_box, selected_anntokens))
    # else:
    #    boxes = nusc.get_boxes(sample_data_token)
    boxes = pred_anns
    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue
        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def render_sample_data(
        sample_toekn: str,
        with_anns: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        axes_limit: float = 40,
        ax=None,
        nsweeps: int = 1,
        out_path: str = None,
        underlay_map: bool = True,
        use_flat_vehicle_coordinates: bool = True,
        show_lidarseg: bool = False,
        show_lidarseg_legend: bool = False,
        filter_lidarseg_labels=None,
        lidarseg_preds_bin_path: str = None,
        verbose: bool = True,
        show_panoptic: bool = False,
        pred_data=None,
      ) -> None:
    # lidiar_render(sample_toekn, pred_data, out_path=out_path)
    sample = nusc.get('sample', sample_toekn)
    # sample = data['results'][sample_token_list[0]][0]
    cams = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ]
    if ax is None:
        _, ax = plt.subplots(2, 3, figsize=(24, 18))
    j = 0
    for ind, cam in enumerate(cams):
        sample_data_token = sample['data'][cam]

        sd_record = nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        if sensor_modality in ['lidar', 'radar']:
            assert False
        elif sensor_modality == 'camera':
            # Load boxes and image.
            boxes = [NuscenesTrackingBox(
                         'predicted',
                         record['translation'], record['size'], Quaternion(record['rotation']),
                         tracking_id=record['tracking_id'].split('_')[-1]) for record in
                     pred_data['results'][sample_toekn] if record['tracking_score'] > 0.4]

            data_path, boxes_pred, camera_intrinsic = get_predicted_data(sample_data_token,
                                                                         box_vis_level=box_vis_level, pred_anns=boxes)
            # _, boxes_gt, _ = nusc.get_sample_data(sample_data_token, box_vis_level=box_vis_level)
            if ind == 3:
                j += 1
            ind = ind % 3
            data = Image.open(data_path)

            # Show image.
            ax[j, ind].imshow(data)

            # Show boxes.
            if with_anns:
                for box in boxes_pred:
                    # c = np.array(get_color(box.name)) / 255.0
                    track_id = int(box.tracking_id)
                    color = COLOR_MAP[COLOR_KEYS[track_id % len(COLOR_KEYS)]]
                    box.render(ax[j, ind], view=camera_intrinsic, normalize=True, \
                        colors=color, linestyle='dashed', linewidth=1.5, text=False)

            # Limit visible range.
            ax[j, ind].set_xlim(0, data.size[0])
            ax[j, ind].set_ylim(data.size[1], 0)

        else:
            raise ValueError("Error: Unknown sensor modality!")

        ax[j, ind].axis('off')
        ax[j, ind].set_aspect('equal')

    if out_path is not None:
        plt.savefig(out_path+'_camera', bbox_inches='tight', pad_inches=0, dpi=200)
    if verbose:
        plt.show()
    plt.close()


def make_videos(fig_dir, fig_names, video_name, video_dir):
    import imageio
    import os
    import cv2

    fileList = list()
    for name in fig_names:
        fileList.append(os.path.join(fig_dir, name))

    writer = imageio.get_writer(os.path.join(video_dir, video_name), fps=2)
    for im in fileList:
        writer.append_data(cv2.resize(imageio.imread(im), (4000, 2800)))
    writer.close()
    return


def parse_args():
    parser = argparse.ArgumentParser(description='3D Tracking Visualization')
    parser.add_argument('--data_infos_path', type=str, default='./data/nuscenes/tracking_forecasting-mini_infos_val.pkl')
    parser.add_argument('--result', help='results file')
    parser.add_argument(
        '--show-dir', help='directory where visualize results will be saved')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    data_infos = mmcv.load(args.data_infos_path)['infos']
    data_info_sample_tokens = [info['token'] for info in data_infos]

    nusc = NuScenes(version='v1.0-trainval', dataroot='../data/nuscenes/', verbose=True)
    
    results = mmcv.load(args.result)
    sample_token_list = list(results['results'].keys())

    pbar = tqdm(total=len(sample_token_list))
    for i, sample_token in enumerate(sample_token_list):
        # prepare the directory for visualization
        data_info_idx = data_info_sample_tokens.index(sample_token)
        sample_info = data_infos[data_info_idx]
        scene_token = sample_info['scene_token']
        seq_dir = os.path.join(args.show_dir, scene_token)
        os.makedirs(seq_dir, exist_ok=True)
        out_path = os.path.join(seq_dir, f'{i}')

        # render
        render_sample_data(sample_token, pred_data=results, out_path=out_path)
        pbar.update(1)
    pbar.close()

    print('Making Videos')
    scene_tokens = os.listdir(args.show_dir)
    for video_index, scene_token in enumerate(scene_tokens):
        show_dir = os.path.join(args.show_dir, scene_token)
        fig_names = os.listdir(show_dir)
        indexes = sorted([int(fname.split('_')[0]) for fname in fig_names if fname.endswith('png')])
        fig_names = [f'{i}_camera.png' for i in indexes]

        make_videos(show_dir, fig_names, 'video.mp4', show_dir)