# Data Preparation

We have successfully pre-trained and fine-tuned our Mu-MAE on [MMAct](https://mmact19.github.io/2019/)
- The pre-processing of **MMAct** can be summarized into 2 steps:

  1. Download the dataset from [official website](https://mmact19.github.io/2019/).

  2. Generate annotations needed for dataloader ("<path_to_video> <video_class>" in annotations). The annotation usually includes `train.csv`, `val.csv` and `test.csv` ( here `test.csv` is the same as `val.csv`). We **share** our annotation files (train.csv, val.csv, test.csv) via [Google Drive](https://drive.google.com/drive/folders/1cfA-SrPhDB9B8ZckPvnh8D5ysCjD-S_I?usp=share_link). The format of `*.csv` file is like:

     ```
     dataset_root/video_1.mp4  label_1
     dataset_root/video_2.mp4  label_2
     dataset_root/video_3.mp4  label_3
     ...
     dataset_root/video_N.mp4  label_N
     ```

### Note:

1. We use [decord](https://github.com/dmlc/decord) to decode the videos **on the fly** during both pre-training and fine-tuning phases.
