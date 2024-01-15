import os
import numpy as np
from numpy.lib.function_base import disp
import torch
import decord
from PIL import Image
from torchvision import transforms
from random_erasing import RandomErasing
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
import video_transforms as video_transforms 
import volume_transforms as volume_transforms
import pandas as pd

class VideoSensorDataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, data_path, mode='train', clip_len=8,
                 frame_sample_rate=2, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3,args=None):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        self.phone_path = "/your_data_path/acc_phone_clip"
        self.watch_path = "/your_data_path/acc_watch_clip"
        self.gyro_path = "/your_data_path/gyro_clip"
        self.orientation_path = "/your_data_path/orientation_clip"
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        import pandas as pd
        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        self.dataset_samples = list(cleaned.values[:, 0])
        self.label_array = list(cleaned.values[:, 1])

        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))

    def _get_sensor_data_with_video_path(self, clip_path,scale_t=1):
        class_folder, sensor_file_name = self._convert_string(clip_path)
        # phone path
        acc_phone_path = os.path.join(self.phone_path, class_folder, sensor_file_name)
        # watch path
        acc_watch_path = os.path.join(self.watch_path, class_folder, sensor_file_name)
        # gyro path
        gyro_path = os.path.join(self.gyro_path, class_folder, sensor_file_name)
        # orientation path
        orientation_path = os.path.join(self.orientation_path, class_folder, sensor_file_name)
        csv_paths = [acc_phone_path, acc_watch_path, gyro_path, orientation_path]
        all_paths = [clip_path] + csv_paths
        if self.check_paths_availability(all_paths) and self.check_files_empty(csv_paths):
            # video data
            video_data = self.loadvideo_decord(clip_path, sample_rate_scale=scale_t)  # T H W C
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
            return video_data, acc_phone_data,acc_watch_data,gyro_data,orientation_data
        else:
            return [],[],[],[],[]

    def check_paths_availability(self,paths):
        for path in paths:
            if not os.path.exists(path):
                warnings.warn("path {} not exist during training".format(path))
                return False
        return True
    def check_files_empty(self, file_paths):
        for file_path in file_paths:
            if self.is_file_empty(file_path):
                warnings.warn("path file {} is empty during training".format(file_path))
                return False
        return True
    def is_file_empty(self, file_path):
        return os.stat(file_path).st_size == 0

    def min_max_norm(self,dataframe_sensor):
        # apply normalization techniques
        for column in dataframe_sensor.columns:
            dataframe_sensor[column] = (dataframe_sensor[column] - dataframe_sensor[column].min()) / (
                        dataframe_sensor[column].max() - dataframe_sensor[column].min())
        return dataframe_sensor

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

    def linear_select_samples(self, tensor, num_samples):
        _, seq_len, _, _ = tensor.size()

        if num_samples > seq_len:
            print("Number of samples cannot exceed the sequence length.so we enlarge the data")
            print("seq_len:" + str(seq_len))
            repeat_size = int(num_samples/seq_len) + 1
            print(repeat_size)
            tensor = self.repeat_size_size(tensor, repeat_size)


        # Generate linearly spaced indices
        step_size = (seq_len - 1) / (num_samples - 1)
        indices = torch.linspace(0, seq_len - 1, num_samples).long()
        # print(indices)

        # Select the samples based on the indices
        selected_samples = tensor[:, indices, :, :]

        return selected_samples

    def repeat_size_size(self,tensor,repeat_size):
        # Get the original dimensions
        original_size = tensor.size()

        # Double the size along the second dimension
        new_size = list(original_size)
        new_size[1] *= repeat_size

        # Repeat the tensor along the second dimension
        doubled_tensor = tensor.repeat(1, repeat_size, 1, 1)[:new_size[0], :new_size[1], :new_size[2], :new_size[3]]

        return doubled_tensor

    def _convert_string(self, input_string):
        # Split the input string using "/"
        parts = input_string.split("/")

        # Extract the second-to-last and last parts
        if len(parts) >= 2:
            second_last = parts[-2]
            last = parts[-1]
            last_parts = last.split('_')

            # Filter out the parts that do not contain "cam"
            filtered_parts = [last_part for last_part in last_parts if "cam" not in last_part]

            # Join the filtered parts with underscores
            sensor_file_path = '_'.join(filtered_parts)
            sensor_file_path_parts = sensor_file_path.split(".")
            sensor_file_path = sensor_file_path_parts[0] + ".csv"
            return second_last, sensor_file_path
        else:
            # Handle the case where the input string does not contain enough parts
            raise(RuntimeError("video file %s doesn't not have corresponding sensor files. " % (input_string)))


    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args 
            scale_t = 1

            sample = self.dataset_samples[index]
            # multiple sensor data
            buffer,acc_phone_data,acc_watch_data,gyro_data,orientation_data = self._get_sensor_data_with_video_path(sample,scale_t)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer,acc_phone_data,acc_watch_data,gyro_data,orientation_data = self._get_sensor_data_with_video_path(sample,scale_t)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    item = (new_frames, acc_phone_data,acc_watch_data,gyro_data,orientation_data)
                    label = self.label_array[index]
                    frame_list.append(item)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)
            
            return (buffer,acc_phone_data,acc_watch_data,gyro_data,orientation_data), self.label_array[index], index, {}

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer,acc_phone_data,acc_watch_data,gyro_data,orientation_data = self._get_sensor_data_with_video_path(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer,acc_phone_data,acc_watch_data,gyro_data,orientation_data = self._get_sensor_data_with_video_path(sample)
            buffer = self.data_transform(buffer)
            return (buffer,acc_phone_data,acc_watch_data,gyro_data,orientation_data), self.label_array[index], sample.split("/")[-1].split(".")[0]

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer,acc_phone_data,acc_watch_data,gyro_data,orientation_data = self._get_sensor_data_with_video_path(sample)

            while len(buffer) == 0:
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(\
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer,acc_phone_data,acc_watch_data,gyro_data,orientation_data = self._get_sensor_data_with_video_path(sample)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                 / (self.test_num_crop - 1)
            temporal_step = max(1.0 * (buffer.shape[0] - self.clip_len) \
                                / (self.test_num_segment - 1), 0)
            temporal_start = int(chunk_nb * temporal_step)
            spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                       spatial_start:spatial_start + self.short_side_size, :, :]
            else:
                buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                       :, spatial_start:spatial_start + self.short_side_size, :]

            buffer = self.data_transform(buffer)
            return (buffer,acc_phone_data,acc_watch_data,gyro_data,orientation_data), self.test_label_array[index], sample.split("/")[-1].split(".")[0], \
                   chunk_nb, split_nb
        else:
            raise NameError('mode {} unkown'.format(self.mode))



    def _aug_frame(
        self,
        buffer,
        args,
    ):

        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [
            transforms.ToPILImage()(frame) for frame in buffer
        ]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer) # T C H W
        buffer = buffer.permute(0, 2, 3, 1) # T H W C 
        
        # T H W C 
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True ,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer


    def loadvideo_decord(self, sample, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample

        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                 num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        if self.mode == 'test':
            all_index = [x for x in range(0, len(vr), self.frame_sample_rate)]
            while len(all_index) < self.clip_len:
                all_index.append(all_index[-1])
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

        # handle temporal segments
        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = len(vr) // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len:
                index = np.linspace(0, seg_len, num=seg_len // self.frame_sample_rate)
                index = np.concatenate((index, np.ones(self.clip_len - seg_len // self.frame_sample_rate) * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i*seg_len
            all_index.extend(list(index))

        all_index = all_index[::int(sample_rate_scale)]
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


class VideoSensorMAE(torch.utils.data.Dataset):
    """Load your own video classification dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are three items in each line: (1) video path; (2) video length and (3) video label.
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default None.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default False.
        Whether to use video loader to load data.
    use_decord : bool, default True.
        Whether to use Decord video loader to load data. Otherwise use mmcv video loader.
    transform : function, default None.
        A function that takes data and label and transforms them.
    data_aug : str, default 'v1'.
        Different types of data augmentation auto. Supports v1, v2, v3 and v4.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """
    def __init__(self,
                 root,
                 setting,
                 train=True,
                 test_mode=False,
                 name_pattern='img_%05d.jpg',
                 video_ext='mp4',
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 transform=None,
                 sensor_mask=None,
                 temporal_jitter=False,
                 video_loader=False,
                 use_decord=False,
                 lazy_init=False):

        super(VideoSensorMAE, self).__init__()
        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord
        self.transform = transform
        self.sensor_mask = sensor_mask
        self.lazy_init = lazy_init
        self.phone_path = "/your_data_path/acc_phone_clip"
        self.watch_path = "/your_data_path/acc_watch_clip"
        self.gyro_path = "/your_data_paths/gyro_clip"
        self.orientation_path = "/your_data_path/orientation_clip"


        if not self.lazy_init:
            self.clips = self._make_dataset(root, setting)
            if len(self.clips) == 0:
                raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                                   "Check your data directory (opt.data-dir)."))

    def __getitem__(self, index):

        directory,acc_phone_path,acc_watch_path,gyro_path,orientation_path, target = self.clips[index]
        #video data
        if self.video_loader:
            if '.' in directory.split('/')[-1]:
                # data in the "setting" file already have extension, e.g., demo.mp4
                video_name = directory
            else:
                # data in the "setting" file do not have extension, e.g., demo
                # So we need to provide extension (i.e., .mp4) to complete the file name.
                video_name = '{}.{}'.format(directory, self.video_ext)

            decord_vr = decord.VideoReader(video_name, num_threads=1)
            duration = len(decord_vr)

        segment_indices, skip_offsets = self._sample_train_indices(duration)

        images = self._video_TSN_decord_batch_loader(directory, decord_vr, duration, segment_indices, skip_offsets)

        vid_data, vid_mask = self.transform((images, None)) # T*C,H,W
        vid_data = vid_data.view((self.new_length, 3) + vid_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
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
        acc_phone_mask, acc_watch_mask, gyro_mask, ori_mask = self.sensor_mask()
        return (vid_data, vid_mask,acc_phone_data,acc_watch_data,gyro_data,orientation_data,acc_phone_mask,acc_watch_mask,gyro_mask,ori_mask)

    def __len__(self):
        return len(self.clips)

    def _convert_string(self, input_string):
        # Split the input string using "/"
        parts = input_string.split("/")

        # Extract the second-to-last and last parts
        if len(parts) >= 2:
            second_last = parts[-2]
            last = parts[-1]
            last_parts = last.split('_')

            # Filter out the parts that do not contain "cam"
            filtered_parts = [last_part for last_part in last_parts if "cam" not in last_part]

            # Join the filtered parts with underscores
            sensor_file_path = '_'.join(filtered_parts)
            sensor_file_path_parts = sensor_file_path.split(".")
            sensor_file_path = sensor_file_path_parts[0] + ".csv"
            return second_last, sensor_file_path
        else:
            # Handle the case where the input string does not contain enough parts
            raise(RuntimeError("video file %s doesn't not have corresponding sensor files. " % (input_string)))
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
            print("Number of samples cannot exceed the sequence length.so we enlarge the data")
            print("seq_len:" + str(seq_len))
            repeat_size = int(num_samples/seq_len) + 1
            print(repeat_size)
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

    def _make_dataset(self, directory, setting):
        if not os.path.exists(setting):
            raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(' ')
                # line format: video_path, video_duration, video_label
                if len(line_info) < 2:
                    raise(RuntimeError('Video input format is not correct, missing one or more element. %s' % line))
                clip_path = os.path.join(line_info[0])
                target = int(line_info[1])
                class_folder, sensor_file_name = self._convert_string(clip_path)
                # phone data
                acc_phone_path = os.path.join(self.phone_path, class_folder, sensor_file_name)
                # watch data
                acc_watch_path = os.path.join(self.watch_path, class_folder, sensor_file_name)
                # gyro data
                gyro_path = os.path.join(self.gyro_path, class_folder, sensor_file_name)
                # orientation data
                orientation_path = os.path.join(self.orientation_path, class_folder, sensor_file_name)
                csv_paths = [acc_phone_path, acc_watch_path, gyro_path, orientation_path]
                all_paths = [clip_path] + csv_paths
                # video data
                if self.check_paths_availability(all_paths) and self.check_files_empty(csv_paths):
                    item = (clip_path,acc_phone_path,acc_watch_path,gyro_path,orientation_path, target)
                    clips.append(item)
        return clips
    def check_paths_availability(self,paths):
        for path in paths:
            if not os.path.exists(path):
                return False
        return True
    def check_files_empty(self, file_paths):
        for file_path in file_paths:
            if self.is_file_empty(file_path):
                return False
        return True
    def is_file_empty(self, file_path):
        return os.stat(file_path).st_size == 0
    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration)
            offsets = offsets + np.random.randint(average_duration,
                                                  size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(np.random.randint(
                num_frames - self.skip_length + 1,
                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets


    def _video_TSN_decord_batch_loader(self, directory, video_reader, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory, duration))
        return sampled_list
