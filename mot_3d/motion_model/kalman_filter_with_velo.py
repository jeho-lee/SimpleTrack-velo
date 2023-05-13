""" Many parts are borrowed from https://github.com/xinshuoweng/AB3DMOT
"""
import numpy as np
from simpletrack.mot_3d.data_protos import BBox
from filterpy.kalman import KalmanFilter

class KalmanFilterVeloMotionModel:
    def __init__(self, bbox: BBox, velo, inst_type, time_stamp, covariance='default'):
        # the time stamp of last observation
        self.prev_time_stamp = time_stamp
        self.latest_time_stamp = time_stamp
        self.score = bbox.s
        self.inst_type = inst_type

        # Define Kalman Filter
        self.kf = KalmanFilter(dim_x=10, dim_z=9)
        
        # State vector [10X1], [x, y, z, theta, l, w, h, vx, vy, vz]   # vz is not used
        self.kf.x[:7] = BBox.bbox2array(bbox)[:7].reshape((7, 1))  # bbox attributes
        self.kf.x[7:9] = np.asarray(velo).reshape((2, 1))  # velocity
        
        # State transition matrix [10X10]
        self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],
                              [0,1,0,0,0,0,0,0,1,0],
                              [0,0,1,0,0,0,0,0,0,1],
                              [0,0,0,1,0,0,0,0,0,0],  
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,1]])

        # Measurement matrix [9X10]: Vz is not observed
        self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0,0,0,0],
                              [0,0,1,0,0,0,0,0,0,0],
                              [0,0,0,1,0,0,0,0,0,0],
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,1,0]])

        self.kf.B = np.zeros((10, 1))                     # dummy control transition matrix
        
        self.covariance_type = covariance
        
        # Define uncertainty matrix
        # self.kf.R[0:,0:] *= 10.   # measurement uncertainty
        # self.kf.P[7:, 7:] *= 1000. # state uncertainty, 
                                    # give high uncertainty to the unobservable initial velocities, covariance matrix
                                    # Here, object velo prediction is availale, hence no need to give high uncertainty
        self.kf.P *= 10.
        # self.kf.Q[7:, 7:] *= 0.01

        # Only used for read the history of the tracklet (bbox, velo)
        self.history = [bbox]
        self.history_velo = [velo]
    
    def get_prediction(self, time_stamp=None):      # from kalman_filter.py
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        time_lag = time_stamp - self.prev_time_stamp
        self.latest_time_stamp = time_stamp
        
        # state transition matrix in current frame
        self.kf.F = np.array([[1,0,0,0,0,0,0,time_lag,0,0],      
                              [0,1,0,0,0,0,0,0,time_lag,0],
                              [0,0,1,0,0,0,0,0,0,time_lag], # Vz is not used
                              [0,0,0,1,0,0,0,0,0,0],
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,1]])
        
        # State predictions of tracklets
        # BBox location (x, y, z) and its velocity is used for prediction
        pred_x = self.kf.get_prediction()[0] # (x, P)
        if pred_x[3] >= np.pi: pred_x[3] -= np.pi * 2
        if pred_x[3] < -np.pi: pred_x[3] += np.pi * 2
        pred_bbox = BBox.array2bbox(pred_x[:7].reshape(-1))
        pred_velo = pred_x[7:9].reshape(-1)

        self.history.append(pred_bbox)
        self.history_velo.append(pred_velo)
        return pred_bbox

    def predict(self, time_stamp=None):
        """ For the motion prediction, use the get_prediction function.
        """
        self.kf.predict()
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
        return

    def update(self, det_bbox: BBox, aux_info):
        """ 
        Updates the state vector with observed bbox and velocity.
        """
        # bbox = BBox.bbox2array(det_bbox)[:7].tolist()
        
        # Observed bbox and velocity
        bbox = BBox.bbox2array(det_bbox)[:7] # KalmanFilter.py
        velo = aux_info['velo']
        
        # full pipeline of kf, first predict, then update
        self.predict()
        
        ######################### orientation correction
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        new_theta = bbox[3]
        if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
        if new_theta < -np.pi: new_theta += np.pi * 2
        bbox[3] = new_theta

        predicted_theta = self.kf.x[3]
        if np.abs(new_theta - predicted_theta) > np.pi / 2.0 and np.abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
            self.kf.x[3] += np.pi       
            if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
            if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if np.abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0: self.kf.x[3] += np.pi * 2
            else: self.kf.x[3] -= np.pi * 2

        #########################     # flip

        # Update the object state
        self.kf.update(np.concatenate((bbox, velo), axis=0))
        
        self.prev_time_stamp = self.latest_time_stamp

        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the rage
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        if det_bbox.s is None:
            self.score = self.score * 0.1
        else:
            self.score = det_bbox.s
        
        cur_bbox = self.kf.x[:7].reshape(-1).tolist()
        cur_bbox = BBox.array2bbox(cur_bbox + [self.score])
        self.history[-1] = cur_bbox
        self.history_velo[-1] = velo
        return

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        cur_bbox = self.kf.x[:7].reshape(-1).tolist()
        return BBox.array2bbox(cur_bbox + [self.score])
    
    def compute_innovation_matrix(self):
        """ compute the innovation matrix for association with mahalonobis distance
        """
        return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R
    
    def sync_time_stamp(self, time_stamp):
        self.time_stamp = time_stamp
        return