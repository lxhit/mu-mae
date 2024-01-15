import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import zipfile
import io
import numpy as np
import random
import re
import pickle
from glob import glob
import pandas as pd

from videotransforms.video_transforms import Compose, Resize, RandomCrop, RandomRotation, ColorJitter, RandomHorizontalFlip, CenterCrop, TenCrop,Rotation
from videotransforms.volume_transforms import ClipToTensor

"""Contains video frame paths and ground truth labels for a single split (e.g. train videos). """
class Split():
    def __init__(self):
        self.gt_a_list = []
        self.videos = []
        self.acc_phone = []
        self.acc_watch = []
        self.gyro = []
        self.orientation = []

    
    def add_multi_data(self, vid_paths,acc_phone_path,acc_watch_path,gyro_path,orientation_path, gt_a):
        self.videos.append(vid_paths)
        self.acc_phone.append(acc_phone_path)
        self.acc_watch.append(acc_watch_path)
        self.gyro.append(gyro_path)
        self.orientation.append(orientation_path)
        self.gt_a_list.append(gt_a)

    def get_rand_multidata(self, label, idx=-1):
        match_idxs = []
        for i in range(len(self.gt_a_list)):
            if label == self.gt_a_list[i]:
                match_idxs.append(i)
        
        if idx != -1:
            return self.videos[match_idxs[idx]],self.acc_phone[match_idxs[idx]],self.acc_watch[match_idxs[idx]],self.gyro[match_idxs[idx]],self.orientation[match_idxs[idx]], match_idxs[idx]
        random_idx = np.random.choice(match_idxs)
        return self.videos[random_idx],self.acc_phone[match_idxs[random_idx]],self.acc_watch[match_idxs[random_idx]],self.gyro[match_idxs[random_idx]], self.orientation[match_idxs[idx]],random_idx

    def get_num_videos_for_class(self, label):
        return len([gt for gt in self.gt_a_list if gt == label])

    def get_unique_classes(self):
        return list(set(self.gt_a_list))

    def __len__(self):
        return len(self.gt_a_list)

"""Dataset for few-shot videos, which returns few-shot tasks. """
class MultisensorDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args

        self.vid_data_dir = args.vid_path
        self.phone_data_dir = args.phone_path
        self.watch_data_dir = args.watch_path
        self.gyro_data_dir = args.gyro_path
        self.orientation_data_dir = args.orientation_path
        self.seq_len = args.seq_len
        self.train = True
        self.tensor_transform = transforms.ToTensor()
        self.img_size = args.img_size

        self.annotation_path = args.traintestlist

        self.way=args.way
        self.shot=args.shot
        self.query_per_class=args.query_per_class

        self.train_split = Split()
        self.test_split = Split()

        self.setup_transforms()
        self._select_fold()
        self.read_dir()

    """Setup crop sizes/flips for augmentation during training and centre crop for testing"""
    def setup_transforms(self):
        # print(self.img_size)
        video_transform_list = []
        video_test_list = []
        video_rotation_list = []
        if self.img_size == 84:
            video_transform_list.append(Resize(96))
            video_test_list.append(Resize(96))
        elif self.img_size == 224:
            video_rotation_list.append(Resize(256))
            video_transform_list.append(Resize(256))
            video_test_list.append(Resize(256))
            video_transform_list.append(RandomHorizontalFlip())
            video_transform_list.append(CenterCrop(self.img_size))
            video_test_list.append(CenterCrop(self.img_size))
        elif self.img_size == 160:
            video_rotation_list.append(Resize(256))
            video_transform_list.append(Resize(256))
            video_test_list.append(Resize(256))
            video_transform_list.append(RandomHorizontalFlip())
            video_transform_list.append(CenterCrop(self.img_size))
            video_test_list.append(CenterCrop(self.img_size))

        else:
            print("img size transforms not setup")
            exit(1)


        video_test_list.append(CenterCrop(self.img_size))
        # video_rotation_list.append(RandomRotation(self.degree))
        #
        # video_rotation_list.append(CenterCrop(self.img_size))
        self.transform = {}
        self.transform["train"] = Compose(video_transform_list)
        self.transform["test"] = Compose(video_test_list)
        self.transform["rotation"] = Compose(video_rotation_list)

    def check_paths_availability(self,paths):
        for path in paths:
            if not os.path.exists(path):
                return False
        return True

    def is_file_empty(self, file_path):
        return os.stat(file_path).st_size == 0

    def check_files_empty(self, file_paths):
        for file_path in file_paths:
            if self.is_file_empty(file_path):
                return False
        return True

    def read_dir(self):
        class_folders = os.listdir(self.vid_data_dir)
        class_folders.sort()
        self.class_folders = class_folders
        for class_folder in class_folders:
            video_folders = os.listdir(os.path.join(self.vid_data_dir, class_folder))
            video_folders.sort()
            if self.args.debug_loader:
                video_folders = video_folders[0:1]
            for video_folder in video_folders:
                # print(video_folder)
                c = self.get_train_or_test_db(video_folder)
                # print(c)
                if c == None:
                    continue
                # video data
                imgs = os.listdir(os.path.join(self.vid_data_dir, class_folder, video_folder))
                if len(imgs) < self.seq_len:
                    continue
                imgs.sort()
                vid_paths = [os.path.join(self.vid_data_dir, class_folder, video_folder, img) for img in imgs]
                vid_paths.sort()
                # phone data
                sensor_file_name = self.convert_string(video_folder)
                acc_phone_path = os.path.join(self.phone_data_dir, class_folder, sensor_file_name)
                # watch data
                acc_watch_path = os.path.join(self.watch_data_dir, class_folder, sensor_file_name)
                # gyro data
                gyro_path = os.path.join(self.gyro_data_dir, class_folder, sensor_file_name)
                # orientation data
                orientation_path = os.path.join(self.orientation_data_dir, class_folder, sensor_file_name)
                # class label
                class_id =  class_folders.index(class_folder)
                csv_paths = [acc_phone_path,acc_watch_path,gyro_path,orientation_path]
                all_paths = vid_paths + csv_paths
                if self.check_paths_availability(all_paths) and self.check_files_empty(csv_paths):
                    c.add_multi_data(vid_paths,acc_phone_path,acc_watch_path,gyro_path,orientation_path, class_id)
        print("loaded {}".format(self.vid_data_dir))
        print("train: {}, test: {}".format(len(self.train_split), len(self.test_split)))

    """ return the current split being used """
    def get_train_or_test_db(self, split=None):
        if split is None:
            get_train_split = self.train
        else:
            if split in self.train_test_lists["train"]:
                get_train_split = True
            elif split in self.train_test_lists["test"]:
                get_train_split = False
            else:
                return None
        if get_train_split:
            return self.train_split
        else:
            return self.test_split
    
    """ load the paths of all videos in the train and test splits. """ 
    def _select_fold(self):
        lists = {}
        for name in ["train", "test"]:
            fname = "{}list{:02d}.txt".format(name, self.args.split)
            f = os.path.join(self.annotation_path, fname)
            selected_files = []
            with open(f, "r") as fid:
                data = fid.readlines()
                data = [x.replace(' ', '_') for x in data]
                data = [x.strip().split(" ")[0] for x in data]
                data = [os.path.splitext(os.path.split(x)[1])[0] for x in data]
                # if "kinetics" in self.args.path:
                #     data = [x[0:11] for x in data]
                selected_files.extend(data)
            lists[name] = selected_files
        self.train_test_lists = lists

    """ Set len to large number as we use lots of random tasks. Stopping point controlled in run.py. """
    def __len__(self):
        c = self.get_train_or_test_db()
        return 1000000
        return len(c)
   
    """ Get the classes used for the current split """
    def get_split_class_list(self):
        c = self.get_train_or_test_db()
        classes = list(set(c.gt_a_list))
        classes.sort()
        return classes
    
    """Loads a single image from a specified path """
    def read_single_image(self, path):
        with Image.open(path) as i:
            i.load()
            return i

    def get_linspace_frames(self,n_frames):
        # monotone
        if n_frames == self.args.seq_len:
            idxs = [int(f) for f in range(n_frames)]
        else:
            excess_frames = n_frames - self.seq_len
            excess_pad = int(min(5, excess_frames / 2))
            if excess_pad < 1:
                start = 0
                end = n_frames - 1
            else:
                start = random.randint(0, excess_pad)
                end = random.randint(n_frames-1 -excess_pad, n_frames-1)


            idx_f = np.linspace(start, end, num=self.seq_len)
            idxs = [int(f) for f in idx_f]
        return idxs

    # removing the "_camX" portion for sensor file path
    def convert_string(self, input_string):
        # Split the input string by underscores
        parts = input_string.split('_')

        # Filter out the parts that do not contain "cam"
        filtered_parts = [part for part in parts if "cam" not in part]

        # Join the filtered parts with underscores
        result = '_'.join(filtered_parts)
        result = result + ".csv"

        return result

    def min_max_norm(self,dataframe_sensor):
        # apply normalization techniques
        for column in dataframe_sensor.columns:
            dataframe_sensor[column] = (dataframe_sensor[column] - dataframe_sensor[column].min()) / (
                        dataframe_sensor[column].max() - dataframe_sensor[column].min())
        return dataframe_sensor

    def repeat_size_size(self,tensor,repeat_size):
        # Get the original dimensions
        original_size = tensor.size()

        # Double the size along the second dimension
        new_size = list(original_size)
        new_size[1] *= repeat_size

        # Repeat the tensor along the second dimension
        doubled_tensor = tensor.repeat(1, repeat_size, 1, 1)[:new_size[0], :new_size[1], :new_size[2], :new_size[3]]

        return doubled_tensor
    def linear_select_samples(self, tensor, num_samples):
        _, seq_len, _, _ = tensor.size()

        if num_samples > seq_len:
            # print("Number of samples cannot exceed the sequence length.so we enlarge the data")
            # print("seq_len:" + str(seq_len))
            repeat_size = int(num_samples/seq_len) + 1
            # print(repeat_size)
            tensor = self.repeat_size_size(tensor, repeat_size)


        # Generate linearly spaced indices
        step_size = (seq_len - 1) / (num_samples - 1)
        indices = torch.linspace(0, seq_len - 1, num_samples).long()
        # print(indices)

        # Select the samples based on the indices
        selected_samples = tensor[:, indices, :, :]

        return selected_samples

    def convert_to_tensor(self,dataframe_sensor):
        # Convert the data to PyTorch tensors
        # Check if any elements have the numpy.object_ data type
        # Convert the array to torch.tensor with dtype=torch.float32
        arr = torch.tensor(dataframe_sensor.values, dtype=torch.float)
        # print("*******************")
        # print(x.size())
        # Reshape the data to have a channel dimension(number_sample, input_channel(1), dim(x,y,z))
        # print("^^^^^^^^^^^^^^^^^^^^^^")
        arr = arr.unsqueeze(1)
        # print(x.size())
        arr = arr.unsqueeze(0)
        arr = self.linear_select_samples(arr,128) #sensor number size
        return arr

    """Gets a single video sequence. Handles sampling if there are more frames than specified. """
    def get_seq(self, label, idx=-1):
        c = self.get_train_or_test_db()
        vid_paths, acc_phone_path, acc_watch_path, gyro_path, orientation_path, vid_id = c.get_rand_multidata(label, idx)
        # video data
        n_frames = len(vid_paths)
        idxs = self.get_linspace_frames(n_frames)
        imgs_orignal = [self.read_single_image(vid_paths[i]) for i in idxs]
        if (self.transform is not None):
            if self.train:
                transform = self.transform["train"]
            else:
                transform = self.transform["test"]

            imgs = [self.tensor_transform(v) for v in transform(imgs_orignal)]

            imgs = torch.stack(imgs)
            # print("imgs_size: %s" % (str(imgs.size())))
            imgs = imgs.permute(1,0,2,3).contiguous()
            # print("imgs_size: %s" % (str(imgs.size())))
            imgs = imgs.unsqueeze(0)
            # print("imgs_size: %s" % (str(imgs.size())))
        # phone data
        acc_phone_data = pd.read_csv(acc_phone_path, header=None, usecols=[1, 2, 3], names=['x', 'y', 'z'])
        acc_phone_data = self.min_max_norm(acc_phone_data)
        acc_phone_data = self.convert_to_tensor(acc_phone_data)
        # watch data
        acc_watch_data = pd.read_csv(acc_watch_path, header=None, usecols=[1, 2, 3], names=['x', 'y', 'z'])
        acc_watch_data = self.min_max_norm(acc_watch_data)
        acc_watch_data = self.convert_to_tensor(acc_watch_data)
        # gyro data
        gyro_data = pd.read_csv(gyro_path, header=None, usecols=[1, 2, 3], names=['x', 'y', 'z'])
        gyro_data = self.min_max_norm(gyro_data)
        gyro_data = self.convert_to_tensor(gyro_data)
        # orientation data
        orientation_data = pd.read_csv(orientation_path, header=None, usecols=[1, 2, 3], names=['x', 'y', 'z'])
        orientation_data = self.min_max_norm(orientation_data)
        orientation_data = self.convert_to_tensor(orientation_data)


        return imgs,acc_phone_data,acc_watch_data,gyro_data,orientation_data,vid_id




    """returns dict of support and target images and labels"""
    def __getitem__(self, index):

        #select classes to use for this task
        c = self.get_train_or_test_db()
        classes = c.get_unique_classes()
        # print("classes : %s" % (str(classes)))
        batch_classes = random.sample(classes, self.way)
        # print("batch_classes : %s" % (str(batch_classes)))

        if self.train:
            n_queries = self.args.query_per_class
        else:
            n_queries = self.args.query_per_class_test

        support_vid_set = []
        support_phone_set = []
        support_watch_set = []
        support_gyro_set = []
        support_orientation_set = []
        support_labels = []

        target_vid_set = []
        target_phone_set = []
        target_watch_set = []
        target_gyro_set = []
        target_orientation_set = []
        target_labels = []
        real_support_labels = []
        real_target_labels = []
        support_pace_set = []
        support_pace_labels = []
        traget_label_flag = random.randint(0,self.way-1)
        for bl, bc in enumerate(batch_classes):
            
            #select shots from the chosen classes
            n_total = c.get_num_videos_for_class(bc)
            idxs = random.sample([i for i in range(n_total)], self.args.shot + n_queries)
            # print("**************")
            # print(idxs)
            for idx in idxs[0:self.args.shot]:
                all_clips,acc_phone_data,acc_watch_data,gyro_data,orientation_data, all_clips_label = self.get_seq(bc, idx)
                support_vid_set.append(all_clips)
                support_phone_set.append(acc_phone_data)
                support_watch_set.append(acc_watch_data)
                support_gyro_set.append(gyro_data)
                support_orientation_set.append(orientation_data)
                support_labels.append(bl)
            if bl == traget_label_flag:
                for idx in idxs[self.args.shot:]:
                    all_clips, acc_phone_data,acc_watch_data,gyro_data,orientation_data,all_clips_label = self.get_seq(bc, idx)
                    # print(vid.size())
                    # print(rot_vid.size())
                    target_vid_set.append(all_clips)
                    target_phone_set.append(acc_phone_data)
                    target_watch_set.append(acc_watch_data)
                    target_gyro_set.append(gyro_data)
                    target_orientation_set.append(orientation_data)
                    target_labels.append(bl)
                    real_target_labels.append(bc)


        s = list(zip(support_vid_set, support_phone_set,support_watch_set,support_gyro_set,support_orientation_set,support_labels))
        random.shuffle(s)
        support_vid_set, support_phone_set,support_watch_set,support_gyro_set,support_orientation_set,support_labels = zip(*s)


        t = list(zip(target_vid_set, target_phone_set,target_watch_set,target_gyro_set,target_orientation_set, target_labels, real_target_labels))
        random.shuffle(t)
        target_vid_set, target_phone_set,target_watch_set,target_gyro_set,target_orientation_set, target_labels, real_target_labels = zip(*t)
        
        support_vid_set = torch.cat(support_vid_set)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$")
        # print("support_vid_set_size %s" % (str(support_vid_set.size())))
        support_phone_set = torch.cat(support_phone_set)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$")
        # print("support_phone_set_size %s" % (str(support_phone_set.size())))
        support_watch_set = torch.cat(support_watch_set)
        support_gyro_set = torch.cat(support_gyro_set)
        support_orientation_set = torch.cat(support_orientation_set)


        target_vid_set = torch.cat(target_vid_set)
        # print("target_set_size %s" % (str(target_set.size())))
        target_phone_set = torch.cat(target_phone_set)
        target_watch_set = torch.cat(target_watch_set)
        target_gyro_set = torch.cat(target_gyro_set)
        target_orientation_set = torch.cat(target_orientation_set)

        support_labels = torch.FloatTensor(support_labels)
        # print("support_labels %s" % (str(support_labels)))
        target_labels = torch.FloatTensor(target_labels)
        # print("target_labels %s" % (str(target_labels)))
        real_target_labels = torch.FloatTensor(real_target_labels)
        # print("real_target_labels %s" % (str(real_target_labels)))
        # batch_classes = torch.FloatTensor(batch_classes)
        # support_pace_set = torch.cat(support_pace_set)
        # support_pace_labels = torch.cat(support_pace_labels)
        # return {"support_set":support_set, "support_labels":support_labels, "target_set":target_set, "target_labels":target_labels, "real_target_labels":real_target_labels, "batch_class_list": batch_classes,"support_pace_set":support_pace_set,"support_pace_labels":support_pace_labels}
        return {
            "support_vid_set":support_vid_set,
            "support_phone_set": support_phone_set,
            "support_watch_set": support_watch_set,
            "support_gyro_set": support_gyro_set,
            "support_orientation_set": support_orientation_set,
            "support_labels":support_labels,
            "target_vid_set":target_vid_set,
            "target_phone_set": target_phone_set,
            "target_watch_set": target_watch_set,
            "target_gyro_set": target_gyro_set,
            "target_orientation_set": target_orientation_set,
            "target_labels":target_labels,
            "real_target_labels":real_target_labels}



