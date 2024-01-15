import numpy as np
import numpy as np
import torch
class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size # 8, 14, 14
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask
class SensorMaskingGenerator:
    def __init__(self, args):
        self.sensor_batch_size = 16
        self.sensor_mask_type = args.sensor_mask_type
        self.tube_sensor_mask_ratio = args.sensor_mask_ratio
        self.acc_phone_mask_ratio = args.sensor_mask_ratio
        self.acc_watch_mask_ratio = args.sensor_mask_ratio
        self.gyro_mask_ratio = args.sensor_mask_ratio
        self.ori_mask_ratio = args.sensor_mask_ratio
        self.tube_mask_num = int(self.sensor_batch_size * self.tube_sensor_mask_ratio)
        self.acc_phone_mask_num = int(self.sensor_batch_size * self.acc_phone_mask_ratio)
        self.acc_watch_mask_num = int(self.sensor_batch_size * self.acc_watch_mask_ratio)
        self.gyro_mask_num = int(self.sensor_batch_size * self.gyro_mask_ratio)
        self.ori_mask_num = int(self.sensor_batch_size * self.ori_mask_ratio)

    def __repr__(self):
        repr_str = "Sensor Maks: total sensor_batch_size {}, acc phone mask ratio {}, acc watch mask ratio {}, gyo mask ratio {}, ori mask ratio {}".format(
            self.sensor_batch_size, self.acc_phone_mask_ratio,self.acc_watch_mask_ratio, self.gyo_mask_ratio, self.ori_mask_ratio
        )
        return repr_str

    def __call__(self):
        if self.sensor_mask_type == 'tube':
            mask = np.hstack([
                np.zeros(self.sensor_batch_size - self.tube_mask_num),
                np.ones(self.tube_mask_num)
            ])
            np.random.shuffle(mask)
            return mask,mask,mask,mask
        else:
            acc_phone_mask = np.hstack([
                np.zeros(self.sensor_batch_size - self.acc_phone_mask_num),
                np.ones(self.acc_phone_mask_num)
            ])
            acc_watch_mask = np.hstack([
                np.zeros(self.sensor_batch_size - self.acc_watch_mask_num),
                np.ones(self.acc_watch_mask_num)
            ])
            gyro_mask = np.hstack([
                np.zeros(self.sensor_batch_size - self.gyro_mask_num),
                np.ones(self.gyro_mask_num)
            ])
            ori_mask = np.hstack([
                np.zeros(self.sensor_batch_size - self.ori_mask_num),
                np.ones(self.ori_mask_num)
            ])
            np.random.shuffle(acc_phone_mask)
            np.random.shuffle(acc_watch_mask)
            np.random.shuffle(gyro_mask)
            np.random.shuffle(ori_mask)
            return acc_phone_mask,acc_watch_mask,gyro_mask,ori_mask