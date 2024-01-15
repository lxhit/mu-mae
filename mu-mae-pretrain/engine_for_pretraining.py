import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, loss_hyper: float = 0.5):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        #  video and other sensor data here
        videos, bool_masked_pos, acc_phone_data,acc_watch_data,gyro_data,orientation_data,acc_phone_mask,acc_watch_mask,gyro_mask,ori_mask = batch
        # video part
        videos = videos.to(device, non_blocking=True)
        acc_phone_data = acc_phone_data.to(device, non_blocking=True)
        acc_watch_data = acc_watch_data.to(device, non_blocking=True)
        gyro_data = gyro_data.to(device, non_blocking=True)
        orientation_data = orientation_data.to(device, non_blocking=True)

        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        acc_phone_mask = acc_phone_mask.to(device, non_blocking=True).flatten(1).to(torch.bool)
        acc_watch_mask = acc_watch_mask.to(device, non_blocking=True).flatten(1).to(torch.bool)
        gyro_mask = gyro_mask.to(device, non_blocking=True).flatten(1).to(torch.bool)
        ori_mask = ori_mask.to(device, non_blocking=True).flatten(1).to(torch.bool)


        # print("videos in train one epoch function: " + str(videos.shape))
        # print("bool_masked_pos in train one epoch function: " + str(bool_masked_pos.shape))
        # print("bool_masked_pos type in train one epoch function: " + str(bool_masked_pos.type()))
        # print("acc_phone_data in train one epoch function: " + str(acc_phone_data.shape))
        # print("acc_phone_mask in train one epoch function: " + str(acc_phone_mask.shape))
        # print("acc_phone_mask type in train one epoch function: " + str(acc_phone_mask.type()))
        # videos in train one epoch function: torch.Size([32, 3, 16, 224, 224])
        # bool_masked_pos in train one epoch function: torch.Size([32, 1568])
        # videos_patch in train one epoch function: torch.Size([32, 1568, 1536])
        # labels in train one epoch function: torch.Size([32, 1408, 1536])
        # acc_phone_data in train one epoch function: torch.Size([32, 1, 128, 1, 3])
        # acc_phone_mask in train one epoch function: torch.Size([32, 16])
        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
            unnorm_videos = videos * std + mean  # in [0, 1]

            if normlize_target:
                videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
            else:
                videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)

            B, _, C = videos_patch.shape
            # print("videos_patch in train one epoch function: " + str(videos_patch.shape))
            labels = videos_patch[bool_masked_pos].reshape(B, -1, C)
            # print("labels in train one epoch function: " + str(labels.shape))

            # calculate the acc phone sensor predict label
            acc_phone_patch_data,acc_phone_labels = get_label_for_sensor_data(acc_phone_data,acc_phone_mask)
            # print("acc_phone_patch_data in train one epoch function: " + str(acc_phone_patch_data.shape))
            # print("acc phone labels in train one epoch function: " + str(acc_phone_labels.shape))
            # calculate the acc watch sensor predict label
            acc_watch_patch_data,acc_watch_labels = get_label_for_sensor_data(acc_watch_data, acc_watch_mask)
            # print("acc watch labels in train one epoch function: " + str(acc_watch_labels.shape))
            # calculate the gyro sensor predict label
            gyro_patch_data,gyro_labels = get_label_for_sensor_data(gyro_data, gyro_mask)
            # print("gyro labels in train one epoch function: " + str(gyro_labels.shape))
            # calculate the ori sensor predict label
            orientation_patch_data,ori_labels = get_label_for_sensor_data(orientation_data, ori_mask)
            # print("ori labels in train one epoch function: " + str(ori_labels.shape))

        with torch.cuda.amp.autocast():
            vid_output,phone_output,watch_output,gyro_output,ori_output = model(videos, bool_masked_pos,acc_phone_patch_data,acc_watch_patch_data,gyro_patch_data,orientation_patch_data,acc_phone_mask,acc_watch_mask,gyro_mask,ori_mask )
            vid_loss = loss_func(input=vid_output, target=labels)
            phone_loss = loss_func(input=phone_output, target=acc_phone_labels)
            watch_loss = loss_func(input=watch_output, target=acc_watch_labels)
            gyro_loss = loss_func(input=gyro_output, target=gyro_labels)
            ori_loss = loss_func(input=ori_output, target=ori_labels)
            loss = vid_loss + loss_hyper * (phone_loss+watch_loss+gyro_loss+ori_loss)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def get_label_for_sensor_data(sensor_data, sensor_mask):
    # calculate the sensor predict label
    sensor_patch_data = rearrange(sensor_data, 'b c (t p) l s -> b p (t l s c)',
                                  p=16)  # (32,1,128,1,3)--(32,16,8*1*3*1)
    sensor_squeeze = rearrange(sensor_data, 'b c (t p) l s -> b p (t l s) c',
                                  p=16)  # (32,1,128,1,3)--(32,16,8*1*3,1)
    sensor_norm = (sensor_squeeze - sensor_squeeze.mean(dim=-2, keepdim=True)
                      ) / (sensor_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
    sensor_patch = rearrange(sensor_norm, 'b n p c -> b n (p c)')  # (32, 16, 24, 1) -- (32,16,24)
    sensor_B, _, sensor_C = sensor_patch.shape
    sensor_labels = sensor_patch[sensor_mask].reshape(sensor_B, -1, sensor_C)  # (32,12,24) if mask ratio=0.75
    return sensor_patch_data, sensor_labels