""" Example of data loader:
    The data loader has to be an iterator:
    Return a dict of frame data
    Users may create the logic of your own data loader
"""
import os, numpy as np, json
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from simpletrack.mot_3d.data_protos import BBox
import simpletrack.mot_3d.utils as utils
from simpletrack.mot_3d.preprocessing import nms

def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: np.ndarray = np.array([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    tm = np.eye(4)
    rotation = Quaternion(rotation)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))
    return tm


def nu_array2mot_bbox(b):
    nu_box = Box(b[:3], b[3:6], Quaternion(b[6:10]))
    mot_bbox = BBox(
        x=nu_box.center[0], y=nu_box.center[1], z=nu_box.center[2],
        w=nu_box.wlh[0], l=nu_box.wlh[1], h=nu_box.wlh[2],
        o=nu_box.orientation.yaw_pitch_roll[0]
    )
    if len(b) == 11:
        mot_bbox.s = b[-1]
    return mot_bbox

# No point cloud for campus dataset
class CampusMOTDataLoader:
    def __init__(self):
        self.ego_cam = 'CAM_FRONT'
    
    def get_frame_info(self, cur_info):
        """
        Format frame info for tracking
        """
        frame_info = dict() # 얻어와야 하는 데이터: time_stamp, ego, det_types, dets, aux_info{velos, is_key_frame}

        timestamp = cur_info['cams'][self.ego_cam]['timestamp'] # in seconds

        # IMU의 ego pose
        ego2global_translation = cur_info['ego2global_translation']
        ego2global_rotation = cur_info['ego2global_rotation']
        ego2global_rotation = Quaternion(ego2global_rotation)

        ego2global_matrix = np.eye(4)
        ego2global_matrix[:3, :3] = ego2global_rotation.rotation_matrix
        ego2global_matrix[:3, 3] = np.transpose(np.array(ego2global_translation))

        # load point cloud
        # lidar_path = cur_info['lidar_path']
        # point_cloud = np.fromfile(os.path.join(lidar_path), dtype=np.float32)
        # point_cloud = np.reshape(point_cloud, (-1, 5))[:, :4]
        # point_cloud = point_cloud[:, :3]

        # load ego pose
        # lidar2ego_translation = cur_info['lidar2ego_translation']
        # lidar2ego_rotation = cur_info['lidar2ego_rotation']
        # lidar2ego_translation = np.asarray(lidar2ego_translation)
        # lidar2ego_rotation = Quaternion(np.asarray(lidar2ego_rotation))

        # lidar frame -> ego frame
        # point_cloud = np.dot(point_cloud, lidar2ego_rotation.rotation_matrix.T)
        # point_cloud += lidar2ego_translation

        # ego frame -> global frame
        # point_cloud = utils.pc2world(ego2global_matrix, point_cloud)
        
        is_key_frame = True
        
        frame_info['time_stamp'] = timestamp
        frame_info['ego'] = ego2global_matrix
        frame_info['is_key_frame'] = is_key_frame
        frame_info['pc'] = None
        
        return frame_info

    def get_det_info(self, preds):
        """
        For association, format detection info from detector's predictions
        - preds: detection results from BEV detector
        """
        
        det_info = dict()
        
        # @@@ Detection 관련 정보들 @@@
        # SimpleTrack에서는 preprocessing - detection.py 로부터 전처리된 npz 파일을 nuscenes_loader.py에서 또 처리해서 input으로 변환
        bboxes, det_types, velos = [], [], []

        # preprocessing/detection.py
        for pred in preds: # sample_result2bbox_array()
            trans, size, rot, score = pred['translation'], pred['size'], pred['rotation'], pred['detection_score']
            
            bbox = trans + size + rot + [score]
            inst_type = pred['detection_name']
            inst_velo = pred['velocity']
            bboxes += [bbox]
            det_types += [inst_type]
            velos += [inst_velo.tolist()] # annos 생성될때 velo만 tolist() 적용 안됐었음, 여기서 적용

        # data_loader/nuscenes_loader.py, NuScenesLoader.__next__()
        dets = [nu_array2mot_bbox(b) for b in bboxes]
        
        # frame_nms (dets, det_types, velos, thres)
        iou_thres = 0.1
        frame_indexes, det_types = nms(dets, det_types, iou_thres)
        dets = [dets[i] for i in frame_indexes]
        velos = [velos[i] for i in frame_indexes]

        # dets = [BBox.bbox2array(d) for d in dets]
        
        # What we need to association
        det_info['dets'] = dets
        det_info['det_types'] = det_types
        det_info['velos'] = velos
        
        return det_info
        

class NuScenesLoader:
    def __init__(self, configs, type_token, segment_name, data_folder, det_data_folder, start_frame, is_all_classes=False):
        """ initialize with the path to data 
        Args:
            data_folder (str): root path to your data
        """
        self.configs = configs
        self.segment = segment_name
        self.data_loader = data_folder
        self.det_data_folder = det_data_folder
        
        # Decide whether to load all classes or not
        self.type_token = type_token
        self.is_all_classes = is_all_classes

        self.ts_info = json.load(open(os.path.join(data_folder, 'ts_info', '{:}.json'.format(self.segment)), 'r'))
        self.ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(self.segment)), 
            allow_pickle=True)
        self.calib_info = np.load(os.path.join(data_folder, 'calib_info', '{:}.npz'.format(self.segment)),
            allow_pickle=True)
        self.dets = np.load(os.path.join(det_data_folder, 'dets', '{:}.npz'.format(self.segment)),
            allow_pickle=True)
        self.det_type_filter = True
        
        self.use_pc = configs['data_loader']['pc']
        if self.use_pc:
            self.pcs = np.load(os.path.join(data_folder, 'pc', 'raw_pc', '{:}.npz'.format(self.segment)),
                allow_pickle=True)

        self.max_frame = len(self.dets['bboxes'])
        self.cur_frame = start_frame
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.cur_frame >= self.max_frame:
            raise StopIteration

        result = dict()
        result['time_stamp'] = self.ts_info[self.cur_frame] * 1e-6
        ego = self.ego_info[str(self.cur_frame)]
        ego_matrix = transform_matrix(ego[:3], ego[3:])
        result['ego'] = ego_matrix

        bboxes = self.dets['bboxes'][self.cur_frame]
        inst_types = self.dets['types'][self.cur_frame]
        
        # jeho
        if self.is_all_classes:
            frame_bboxes = bboxes
            result['det_types'] = inst_types
        else:
            frame_bboxes = [bboxes[i] for i in range(len(bboxes)) if inst_types[i] in self.type_token]
            result['det_types'] = [inst_types[i] for i in range(len(inst_types)) if inst_types[i] in self.type_token]
        
        result['dets'] = [nu_array2mot_bbox(b) for b in frame_bboxes]
        result['aux_info'] = dict()
        if 'velos' in list(self.dets.keys()):
            cur_velos = self.dets['velos'][self.cur_frame]
            
            # jeho
            if self.is_all_classes:
                result['aux_info']['velos'] = cur_velos
            else:
                result['aux_info']['velos'] = [cur_velos[i] for i in range(len(cur_velos)) 
                    if inst_types[i] in self.type_token]
        else:
            result['aux_info']['velos'] = None

        result['dets'], result['det_types'], result['aux_info']['velos'] = \
            self.frame_nms(result['dets'], result['det_types'], result['aux_info']['velos'], 0.1)
        result['dets'] = [BBox.bbox2array(d) for d in result['dets']]

        result['pc'] = None
        if self.use_pc:
            pc = self.pcs[str(self.cur_frame)][:, :3]
            calib = self.calib_info[str(self.cur_frame)]
            calib_trans, calib_rot = np.asarray(calib[:3]), Quaternion(np.asarray(calib[3:]))
            pc = np.dot(pc, calib_rot.rotation_matrix.T)
            pc += calib_trans
            result['pc'] = utils.pc2world(ego_matrix, pc)
        
        # if 'velos' in list(self.dets.keys()):
        #     cur_frame_velos = self.dets['velos'][self.cur_frame]
        #     result['aux_info']['velos'] = [cur_frame_velos[i] 
        #         for i in range(len(bboxes)) if inst_types[i] in self.type_token]
        result['aux_info']['is_key_frame'] = True

        self.cur_frame += 1
        return result
    
    def __len__(self):
        return self.max_frame
    
    def frame_nms(self, dets, det_types, velos, thres):
        frame_indexes, frame_types = nms(dets, det_types, thres)
        result_dets = [dets[i] for i in frame_indexes]
        result_velos = None
        if velos is not None:
            result_velos = [velos[i] for i in frame_indexes]
        return result_dets, frame_types, result_velos


class NuScenesLoader10Hz:
    def __init__(self, configs, type_token, segment_name, data_folder, det_data_folder, start_frame):
        """ initialize with the path to data 
        Args:
            data_folder (str): root path to your data
        """
        self.configs = configs
        self.segment = segment_name
        self.data_loader = data_folder
        self.det_data_folder = det_data_folder
        self.type_token = type_token

        self.ts_info = json.load(open(os.path.join(data_folder, 'ts_info', '{:}.json'.format(segment_name)), 'r'))
        self.time_stamps = [t[0] for t in self.ts_info]
        self.is_key_frames = [t[1] for t in self.ts_info]

        self.token_info = json.load(open(os.path.join(data_folder, 'token_info', '{:}.json'.format(segment_name)), 'r'))
        self.ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)), 
            allow_pickle=True)
        self.calib_info = np.load(os.path.join(data_folder, 'calib_info', '{:}.npz'.format(segment_name)),
            allow_pickle=True)
        self.dets = np.load(os.path.join(det_data_folder, 'dets', '{:}.npz'.format(segment_name)),
            allow_pickle=True)
        self.det_type_filter = True
        
        self.use_pc = configs['data_loader']['pc']
        if self.use_pc:
            self.pcs = np.load(os.path.join(data_folder, 'pc', 'raw_pc', '{:}.npz'.format(segment_name)),
                allow_pickle=True)

        self.max_frame = len(self.dets['bboxes'])
        self.selected_frames = [i for i in range(self.max_frame) if self.token_info[i][3]]
        self.cur_selected_index = 0
        self.cur_frame = start_frame
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.cur_selected_index >= len(self.selected_frames):
            raise StopIteration
        self.cur_frame = self.selected_frames[self.cur_selected_index]

        result = dict()
        result['time_stamp'] = self.time_stamps[self.cur_frame] * 1e-6
        ego = self.ego_info[str(self.cur_frame)]
        ego_matrix = transform_matrix(ego[:3], ego[3:])
        result['ego'] = ego_matrix

        bboxes = self.dets['bboxes'][self.cur_frame]
        inst_types = self.dets['types'][self.cur_frame]
        frame_bboxes = [bboxes[i] for i in range(len(bboxes)) if inst_types[i] in self.type_token]
        result['det_types'] = [inst_types[i] for i in range(len(inst_types)) if inst_types[i] in self.type_token]
        
        result['dets'] = [nu_array2mot_bbox(b) for b in frame_bboxes]
        result['aux_info'] = dict()
        if 'velos' in list(self.dets.keys()):
            cur_velos = self.dets['velos'][self.cur_frame]
            result['aux_info']['velos'] = [cur_velos[i] for i in range(len(cur_velos)) 
                if inst_types[i] in self.type_token]
        else:
            result['aux_info']['velos'] = None
        result['dets'], result['det_types'], result['aux_info']['velos'] = \
            self.frame_nms(result['dets'], result['det_types'], result['aux_info']['velos'], 0.1)
        result['dets'] = [BBox.bbox2array(d) for d in result['dets']]
        
        result['pc'] = None
        if self.use_pc:
            pc = self.pcs[str(self.cur_frame)][:, :3]
            calib = self.calib_info[str(self.cur_frame)]
            calib_trans, calib_rot = np.asarray(calib[:3]), Quaternion(np.asarray(calib[3:]))
            pc = np.dot(pc, calib_rot.rotation_matrix.T)
            pc += calib_trans
            result['pc'] = utils.pc2world(ego_matrix, pc)
        
        # if 'velos' in list(self.dets.keys()):
        #     cur_frame_velos = self.dets['velos'][self.cur_frame]
        #     result['aux_info']['velos'] = [cur_frame_velos[i] 
        #         for i in range(len(bboxes)) if inst_types[i] in self.type_token]
        #     print(result['aux_info']['velos'])
        result['aux_info']['is_key_frame'] = self.is_key_frames[self.cur_frame]

        self.cur_selected_index += 1
        return result
    
    def __len__(self):
        return len(self.selected_frames)
    
    def frame_nms(self, dets, det_types, velos, thres):
        frame_indexes, frame_types = nms(dets, det_types, thres)
        result_dets = [dets[i] for i in frame_indexes]
        result_velos = None
        if velos is not None:
            result_velos = [velos[i] for i in frame_indexes]
        return result_dets, frame_types, result_velos
