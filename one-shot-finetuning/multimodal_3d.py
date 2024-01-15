import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models import r21d, r3d, c3d, s3d_g,cam_3d,resnet,cam_r3d,i3res,cam_vivit,cam_tsn,tsn_resnet
import copy
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table
from models.video_transformer import TimeSformer
import modeling_mae
from timm.models import create_model
from collections import OrderedDict
from utils import load_state_dict
from modeling_mae import SensorPatchEmbed
from einops import rearrange

def one_hot(labels_train):
    labels_train = labels_train.cpu()
    nKnovel = 5
    labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
    labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
    labels_train_1hot = torch.zeros(labels_train_1hot_size).scatter_(len(labels_train_1hot_size) - 1, labels_train_unsqueeze, 1)
    return labels_train_1hot


class SensorCNN(nn.Module):
	def __init__(self, input_dim, hidden_dim, out_dim):
		super(SensorCNN, self).__init__()
		self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3)
		self.relu = nn.ReLU()
		self.pool = nn.MaxPool1d(kernel_size=2)
		self.fc = nn.Linear(hidden_dim * 63, out_dim)

	def forward(self, x):
		x = x.view(x.size(0), x.size(3), -1)
		x = self.conv1(x)
		x = self.relu(x)
		x = self.pool(x)
		x = x.view(x.size(0), -1)
		# print(x.size())
		x = self.fc(x)
		return x


class CrossAttention(nn.Module):
	def __init__(self, input_dim, num_heads):
		super(CrossAttention, self).__init__()
		self.input_dim = input_dim
		self.num_heads = num_heads
		self.head_dim = input_dim // num_heads

		# Learnable weight matrices for Q, K, and V
		self.W_q = nn.Linear(input_dim, input_dim)
		self.W_k = nn.Linear(input_dim, input_dim)
		self.W_v = nn.Linear(input_dim, input_dim)

		self.softmax = nn.Softmax(dim=-1)

	def forward(self, q,x):
		seq_len, _ = q.size()

		# Apply linear transformations to obtain Q, K, and V
		q = self.W_q(q)
		k = self.W_k(x)
		v = self.W_v(x)

		# Reshape Q, K, and V for multi-head attention
		q = q.view(seq_len, self.num_heads, self.head_dim)
		k = k.view(seq_len, self.num_heads, self.head_dim)
		v = v.view(seq_len, self.num_heads, self.head_dim)

		# Transpose dimensions for matrix multiplication
		q = q.transpose(0, 1)  # [ num_heads, seq_len, head_dim]
		k = k.transpose(0, 1)  # [num_heads, seq_len, head_dim]
		v = v.transpose(0, 1)  # [num_heads, seq_len, head_dim]

		# Calculate attention scores
		scores = torch.matmul(q, k.transpose(-2, -1))  # [num_heads, seq_len, seq_len]
		scores = scores / (self.head_dim ** 0.5)  # Scale by square root of head dimension

		# Apply softmax activation to obtain attention weights
		attention_weights = self.softmax(scores)

		# Apply attention weights to values
		attended_values = torch.matmul(attention_weights, v)  # [batch_size, num_heads, seq_len, head_dim]

		# Transpose dimensions and reshape for output
		attended_values = attended_values.transpose(0, 1)  # [batch_size, seq_len, num_heads, head_dim]
		attended_values = attended_values.reshape(seq_len,
		                                          self.input_dim)  # [batch_size, seq_len, input_dim]

		return attended_values


class FusionModel(nn.Module):
    def __init__(self, video_dim, sensor_dim, fused_dim):
        super(FusionModel, self).__init__()
        self.video_dim = video_dim
        self.sensor_dim = sensor_dim
        self.fused_dim = fused_dim

        # Learnable weights for video and sensor representations
        self.video_weight = nn.Parameter(torch.randn(1, video_dim))
        self.sensor_weights = nn.ParameterList([
            nn.Parameter(torch.randn(1, sensor_dim)) for _ in range(4)
        ])

        # Linear transformation for fused representation
        self.fusion_linear = nn.Linear(video_dim + 4 * sensor_dim, fused_dim)

    def forward(self, video, sensors):
        batch_size = video.size(0)

        # Apply learnable weights to video and sensor representations
        video = video * self.video_weight
        fused_sensors = []
        for i in range(4):
            fused_sensors.append(sensors[i] * self.sensor_weights[i])

        # Concatenate video and sensor representations
        fused_representation = torch.cat([video] + fused_sensors, dim=1)

        # Apply linear transformation for fused representation
        fused_representation = self.fusion_linear(fused_representation)

        return fused_representation

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.scale_cls = args.scale_cls
        ## 2. init model
        if self.args.method == 'r21d':
            self.base = r21d.R2Plus1DNet(num_classes=self.args.train_num_classes)
            self.cam = cam_3d.CAM()
            self.nFeat = 512
        elif self.args.method == 'r3d':
            self.base = resnet.R3D700(pretrained=True,pre_data=self.args.pre_data)
            self.cam = cam_r3d.CAM()
            self.nFeat = 2048
        elif self.args.method == 'c3d':
            self.base = c3d.C3D(num_classes=self.args.train_num_classes)
            self.cam = cam_3d.CAM()
            self.nFeat = 512
        elif self.args.method == 's3d':
            self.base = s3d_g.S3D(num_classes=self.args.train_num_classes, space_to_depth=False)
            self.cam = cam_3d.CAM()
            self.nFeat = 512
        elif self.args.method == 'inr3d':
            resnet_2d = torchvision.models.resnet50(pretrained=True)
            self.base = i3res.I3ResNet(copy.deepcopy(resnet_2d), self.args.seq_len)
            self.cam = cam_r3d.CAM()
            self.nFeat = 2048
        elif self.args.method == 'vivit':
            self.base = TimeSformer(num_frames=self.args.seq_len,
                                pretrained="/your_data/mu-mae-vit-base.pth",
                                img_size=224,
                                patch_size=16,
                                embed_dims=768,
                                in_channels=3,
                                attention_type='divided_space_time',
                                use_learnable_pos_emb=True,
                                return_cls_token=True)

            self.cam = cam_vivit.CAM()
            self.nFeat = 192
        elif self.args.method == 'tsn':
            self.base = tsn_resnet.TSN(num_class=self.args.train_num_classes, num_segments=self.args.seq_len,new_length=1, modality='RGB',
                base_model="resnet50")
            self.cam = cam_tsn.CAM()
            self.nFeat = 128

        self.mae_model = self._load_pretrain_mae()

        # self.clasifier = nn.Conv2d(self.nFeat, self.args.train_num_classes, kernel_size=1)
        # self.score_pool = nn.AdaptiveAvgPool3d(1)
        self.clasifier = nn.Conv3d(self.nFeat, self.args.train_num_classes, kernel_size=1)
        self.CNN_phone =  SensorCNN(3, 16, 64)
        self.CNN_watch = SensorCNN(3, 16, 64)
        self.CNN_gyro = SensorCNN(3, 16, 64)
        self.CNN_orientation = SensorCNN(3, 16, 64)

        # self.embed_phone =  SensorPatchEmbed(24, 64)
        # self.embed_watch = SensorPatchEmbed(24, 64)
        # self.embed_gyro = SensorPatchEmbed(24, 64)
        # self.embed_orientation = SensorPatchEmbed(24, 64)

        self.mae_to_cross_video = nn.Linear(self.mae_model.embed_dim, 768, bias=False)
        self.mae_to_cross_phone = nn.Linear(self.mae_model.embed_dim, 64, bias=False)
        self.mae_to_cross_watch = nn.Linear(self.mae_model.embed_dim, 64, bias=False)
        self.mae_to_cross_gyro = nn.Linear(self.mae_model.embed_dim, 64, bias=False)
        self.mae_to_cross_ori = nn.Linear(self.mae_model.embed_dim, 64, bias=False)

        self.cross_attention_video = CrossAttention(768, 4) # input_dim (vit: 768), num_heads (4)
        self.cross_attention_phone = CrossAttention(64, 4)  # input_dim (1dcnn: 64), num_heads (4)
        self.cross_attention_watch = CrossAttention(64, 4)  # input_dim (1dcnn: 64), num_heads (4)
        self.cross_attention_gyro = CrossAttention(64, 4)  # input_dim (1dcnn: 64), num_heads (4)
        self.cross_attention_orientation = CrossAttention(64, 4)  # input_dim (1dcnn: 64), num_heads (4)

        self.fusion_model = FusionModel(768, 64, 768) # video_dim 768, sensor_dim: 64, fused_dim:768


        # self.linear = nn.Linear(self.nFeat, self.args.train_num_classes)

    def _load_pretrain_mae(self):
        drop_path = 0.1
        init_scale = 0.01
        mae_model = create_model(
            self.args.mae_model_name,
            pretrained=False,
            num_classes=self.args.train_num_classes,
            all_frames=self.args.seq_len,
            drop_path_rate=drop_path,
            init_scale= init_scale,
        )
        if self.args.mae_model_path:
            model_key_str = 'model|module'
            checkpoint = torch.load(self.args.mae_model_path, map_location='cpu')
            print("Load ckpt from %s" % self.args.mae_model_path)
            checkpoint_model = None
            for model_key in model_key_str.split('|'):
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    print("Load state_dict by model_key = %s" % model_key)
                    break
            if checkpoint_model is None:
                checkpoint_model = checkpoint
            state_dict = mae_model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('backbone.'):
                    new_dict[key[9:]] = checkpoint_model[key]
                elif key.startswith('encoder.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    new_dict[key] = checkpoint_model[key]
            checkpoint_model = new_dict

            # interpolate position embedding
            if 'pos_embed' in checkpoint_model:
                pos_embed_checkpoint = checkpoint_model['pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
                num_patches = mae_model.patch_embed.num_patches  #
                num_extra_tokens = mae_model.pos_embed.shape[-2] - num_patches  # 0/1

                # height (== width) for the checkpoint position embedding
                orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (
                            self.args.seq_len // mae_model.patch_embed.tubelet_size)) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int((num_patches // (self.args.seq_len // mae_model.patch_embed.tubelet_size)) ** 0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    # B, L, C -> BT, H, W, C -> BT, C, H, W
                    pos_tokens = pos_tokens.reshape(-1, self.args.seq_len // mae_model.patch_embed.tubelet_size, orig_size,
                                                    orig_size, embedding_size)
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                    # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1,
                                                                        self.args.seq_len // mae_model.patch_embed.tubelet_size,
                                                                        new_size, new_size, embedding_size)
                    pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    checkpoint_model['pos_embed'] = new_pos_embed

            load_state_dict(mae_model, checkpoint_model)
        return mae_model

    def test_cross(self, ftrain, ftest):
        ftrain = ftrain.mean(5)
        # print("ftrain : %s" % (str(ftrain.size())))
        ftrain = ftrain.mean(5)
        # print("ftrain : %s" % (str(ftrain.size())))
        ftrain = ftrain.mean(4)
        # print("ftrain : %s" % (str(ftrain.size())))
        # print("ftest : %s" % (str(ftest.size())))
        ftest = ftest.mean(5)
        # print("ftest : %s" % (str(ftest.size())))
        ftest = ftest.mean(5)
        # print("ftest : %s" % (str(ftest.size())))
        ftest = ftest.mean(4)
        # print("ftest : %s" % (str(ftest.size())))
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        # print("ftest in cross : %s" % (str(ftest.size())))
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        # print("ftrain in cross: %s" % (str(ftrain.size())))
        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        return scores

    def test_self(self, ftrain, ftest):
        ftrain = ftrain.mean(5)
        # print("ftrain : %s" % (str(ftrain.size())))
        ftrain = ftrain.mean(4)
        # print("ftrain : %s" % (str(ftrain.size())))
        ftrain = ftrain.mean(3)
        # print("ftrain : %s" % (str(ftrain.size())))
        ftrain = ftrain.unsqueeze(1)
        # print("ftrain : %s" % (str(ftrain.size())))
        # print("ftest : %s" % (str(ftest.size())))
        ftest = ftest.mean(5)
        # print("ftest : %s" % (str(ftest.size())))
        ftest = ftest.mean(4)
        # print("ftest : %s" % (str(ftest.size())))
        ftest = ftest.mean(3)
        # print("ftest : %s" % (str(ftest.size())))
        ftest = ftest.unsqueeze(2)
        # print("ftest : %s" % (str(ftest.size())))
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        # print("ftest in self : %s" % (str(ftest.size())))
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        # print("ftrain in self : %s" % (str(ftrain.size())))
        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        return scores
    def get_cls_score_cross(self,ftrain,ftest,batch_size,num_test):
        # print(ftrain.size())
        ftrain = ftrain.mean(5)
        # print(ftrain.size())
        ftrain = ftrain.mean(5)
        # print(ftrain.size())
        ftrain = ftrain.mean(4)
        # accuracy_score = self.test(ftrain, ftest)
        # if not self.training:
        #     return self.test(ftrain, ftest)

        ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        # print("11111111111111")
        # print(ftrain_norm.size())
        # print(ftest_norm.size())
        ftrain_norm = ftrain_norm.unsqueeze(4)
        ftrain_norm = ftrain_norm.unsqueeze(5)
        # print("22222222222222")
        # print(ftrain_norm.size())
        ftrain_norm = ftrain_norm.unsqueeze(6)
        # print("3333333333333333")
        # print(ftrain_norm.size())
        cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)
        # print("44444444444444444")
        # print(cls_scores.size())
        cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])
        # print("44444444444444444")
        # print(cls_scores.size())
        return cls_scores
    def get_cls_score_self(self,ftrain,ftest,batch_size,num_test):
        ftrain = ftrain.mean(5)
        ftrain = ftrain.mean(4)
        ftrain = ftrain.mean(3)
        ftrain = ftrain.unsqueeze(1)
        ftest_un = ftest.unsqueeze(2)
        # accuracy_score = self.test(ftrain, ftest)
        # if not self.training:
        #     return self.test(ftrain, ftest)
        ftest_norm = F.normalize(ftest_un, p=2, dim=3, eps=1e-12)
        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        # print("11111111111111")
        # print(ftrain_norm.size())
        # print(ftest_norm.size())
        ftrain_norm = ftrain_norm.unsqueeze(4)
        ftrain_norm = ftrain_norm.unsqueeze(5)
        # print("22222222222222")
        # print(ftrain_norm.size())
        ftrain_norm = ftrain_norm.unsqueeze(6)
        # print("3333333333333333")
        # print(ftrain_norm.size())
        cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)
        # print("44444444444444444")
        # print(cls_scores.size())
        cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])
        # print("44444444444444444")
        # print(cls_scores.size())
        return cls_scores
    def forward(self, x_vid_train, x_phone_train,x_watch_train,x_gyro_train,x_orientation_train,x_vid_test, x_phone_test,x_watch_test,x_gyro_test,x_orientation_test, ytrain, ytest):
        # #

        # print("^^^^^^^^^^^^&^&^&^&^&^&^^^^^^^^^")
        # print(x_vid_train.size())
        # print("^^^^^^^^^^^^&^&^&^&^&^&!!!!^^^^^^^^^")
        # print(x_phone_train.size())

        batch_size, num_train = 1, x_vid_train.size(0)
        num_test = x_vid_test.size(0)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)
        x = torch.cat((x_vid_train, x_vid_test), 0)

        # video representation f_x [train_num+test_num, 768]
        label, vid_repres, vid_repres_view = self.base(x)
        # print("vid_repres shape after embeding in modeling 3d: " + str(vid_repres.shape))
        # phone representation [train_num+test_num, 64]
        phone_x_all = torch.cat((x_phone_train, x_phone_test), 0)
        # print("phone_x_all shape after cat train and test in modeling 3d: " + str(phone_x_all.shape))
        # phone_x = rearrange(phone_x_all, 'b (t p) l s -> b p (t l s)',
        #                               p=16)  # (6,128,1,3)--(6,16,8*1*3)
        phone_repres = self.CNN_phone(phone_x_all)
        # print("phone_repres shape after embed phone in modeling 3d: " + str(phone_repres.shape))
        # watch representation [train_num+test_num, 64]
        watch_x_all = torch.cat((x_watch_train, x_watch_test), 0)
        # watch_x = rearrange(watch_x_all, 'b (t p) l s -> b p (t l s)',
        #                               p=16)  # (32,1,128,1,3)--(32,16,8*1*3*1)
        watch_repres = self.CNN_watch(watch_x_all)
        # gyro representation [train_num+test_num, 64]
        gyro_x_all = torch.cat((x_gyro_train, x_gyro_test), 0)
        # gyro_x = rearrange(gyro_x_all, 'b (t p) l s -> b p (t l s)',
        #                               p=16)  # (32,1,128,1,3)--(32,16,8*1*3*1)
        gyro_repres = self.CNN_gyro(gyro_x_all)
        # orientation representation [train_num+test_num, 64]
        orientation_x_all = torch.cat((x_orientation_train, x_orientation_test), 0)
        # orientation_x = rearrange(orientation_x_all, 'b (t p) l s -> b p (t l s)',
        #                               p=16)  # (32,1,128,1,3)--(32,16,8*1*3*1)
        orientation_repres = self.CNN_orientation(orientation_x_all)

        # mae representation
        mae_rep = self.mae_model(x,phone_x_all,watch_x_all,gyro_x_all,orientation_x_all)
        video_mae_q = self.mae_to_cross_video(mae_rep)
        phone_mae_q = self.mae_to_cross_phone(mae_rep)
        watch_mae_q = self.mae_to_cross_watch(mae_rep)
        gyro_mae_q = self.mae_to_cross_gyro(mae_rep)
        ori_mae_q = self.mae_to_cross_ori(mae_rep)
        # print("video mae q shape in modeling 3d: " + str(video_mae_q.shape))
        # print("phone_mae_q shape in modeling 3d: " + str(phone_mae_q.shape))


        # self attention for each modality
        vid_repres = self.cross_attention_video(video_mae_q,vid_repres)

        phone_repres = self.cross_attention_phone(phone_mae_q,phone_repres)
        watch_repres = self.cross_attention_watch(watch_mae_q,watch_repres)
        gyro_repres = self.cross_attention_gyro(gyro_mae_q,gyro_repres)
        orientation_repres = self.cross_attention_orientation(ori_mae_q,orientation_repres)

        # fusion model for each modality
        fusion_repres = self.fusion_model(vid_repres,[phone_repres,watch_repres,gyro_repres,orientation_repres])

        # # Print the final weights for each sensor data
        # for i, weight in enumerate(fusion_model.sensor_weights):
        #     print(f"Sensor {i + 1} weight: {weight.item()}")

        # conver 768 to 192,1,2,2
        f = fusion_repres.view(fusion_repres.size(0), -1, 1, 2, 2)
        # print(f.size())
        # few shot learning for the final representation
        _,_,d,h,w = f.size()
        # print(label.size())
        ftrain = f[:batch_size * num_train]
        # print("**************")
        # print(ftrain.size())
        ftrain = ftrain.view(batch_size, num_train, -1)
        # print(ftrain.size())
        ftrain = torch.bmm(ytrain, ftrain)
        # print(ftrain.size())
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        # print(ftrain.size())
        ftrain = ftrain.view(batch_size, -1, *f.size()[1:])
        # print(ftrain.size())
        ftest = f[batch_size * num_train:]
        # print("##############")
        # print(ftest.size())
        ftest = ftest.view(batch_size, num_test, *f.size()[1:])
        # print(ftest.size())


        # b, n1, c, d,h, w = ftest.size()
        # ftrain = ftrain.view(batch_size,-1,c*d,h,w)
        # ftest = ftest.view(batch_size,-1,c*d,h,w)
        # print("^^^^^^^^^^^^^^^^^^^^")
        # print("ftrain : %s" % (str(ftrain.size())))
        # print("ftest : %s" % (str(ftest.size())))
        ftrain_self, ftest_self, ftrain_cross, ftest_cross = self.cam(ftrain, ftest)
        # print("&&&&&&&&&&&")
        # print("ftrain_self size : %s" % str(ftrain_self.size()))
        # print("ftest_self size : %s" % str(ftest_self.size()))
        # print("ftrain_cross size : %s" % str(ftrain_cross.size()))
        # print("ftest_cross size : %s" % str(ftest_cross.size()))
        accuracy_score_self = self.test_self(ftrain_self, ftest_self)
        accuracy_score_cross = self.test_cross(ftrain_cross, ftest_cross)
        # print(accuracy_score_self)
        # print(accuracy_score_cross)

        accuracy_score = torch.add(accuracy_score_cross, accuracy_score_self, alpha=self.args.alpha)
        # print(accuracy_score)
        cls_scores_self = self.get_cls_score_self(ftrain_self,ftest_self,batch_size,num_test)
        cls_scores_cross = self.get_cls_score_cross(ftrain_cross,ftest_cross,batch_size,num_test)

        cls_scores = torch.add(cls_scores_cross, cls_scores_self, alpha=self.args.alpha)


        ftest = ftest_cross
        ftest = ftest.view(batch_size, num_test, K, -1)
        ftest = ftest.transpose(2, 3) 
        ytest = ytest.unsqueeze(3)
        # print("555555555555555555555")
        # print(ftest.size())
        # print(ytest.size())
        ftest = torch.matmul(ftest, ytest)
        # print("555555555555555555555")
        # print(ftest.size())
        ftest = ftest.view(batch_size * num_test, -1, d,h, w)
        # print("555555555555555555555")
        # print(ftest.size())
        ytest = self.clasifier(ftest)
        # print(ytest.size())
        return ytest, cls_scores,accuracy_score



    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            # self.base.cuda(self.args.start_gpu)
            self.base = torch.nn.DataParallel(self.base, device_ids=[i for i in range(self.args.start_gpu, self.args.start_gpu+self.args.num_gpus)])
            self.cam = torch.nn.DataParallel(self.cam, device_ids=[i for i in range(self.args.start_gpu, self.args.start_gpu+self.args.num_gpus)])
            # self.score_pool = torch.nn.DataParallel(self.score_pool, device_ids=[i for i in range(self.args.start_gpu, self.args.start_gpu+self.args.num_gpus)])
            self.clasifier = torch.nn.DataParallel(self.clasifier, device_ids=[i for i in range(self.args.start_gpu, self.args.start_gpu+self.args.num_gpus)])
            # self.linear = torch.nn.DataParallel(self.linear, device_ids=[i for i in range(0, self.args.num_gpus)])



if __name__ == '__main__':

    class ArgsObject(object):
        def __init__(self):
            self.trans_linear_in_dim = 512
            self.trans_linear_out_dim = 128
            self.train_num_classes = 100
            self.way = 5
            self.shot = 1
            self.query_per_class = 5
            self.trans_dropout = 0.1
            self.seq_len = 16
            self.img_size = 84
            self.num_gpus = 1
            self.temp_set = [2,3]
            self.scale_cls=7
            # self.method = 'tsn'
            self.method = 'vivit'
            self.alpha = 1
            self.pre_data = "KMS"
    args = ArgsObject()
    torch.manual_seed(0)
    net = Model(args)
    net.eval()
    # x = torch.rand(1,2,5)
    # y = torch.rand(1,2,5)
    # print(x)
    # print(y)
    # z = torch.add(x,y,alpha=10)
    # print(z)
    x1 = torch.rand(15, 3, 16, 224, 224)
    x2 = torch.rand(2, 3, 16, 224, 224)
    y1 = torch.rand(1, 15, 5)
    y2 = torch.rand(1, 2, 5)
    #compute the flops
    # inputs = (x1,x2,y1,y2)
    # flop = FlopCountAnalysis(net, inputs)
    # print(flop_count_table(flop, max_depth=4, show_param_shapes=False))
    # print(flop_count_str(flop))
    # print("Total", flop.total() / 1e9)

    ytest, cls_scores,accuracy_score = net(x1, x2, y1, y2)

    print(cls_scores.size())
    print(accuracy_score.size())
    accuracy_score = accuracy_score.view(1 * 5, -1)
    print(accuracy_score.size())
    _, preds = torch.max(accuracy_score.detach().cpu(), 1)
    print(preds.size())
    print(preds)
    acc = (torch.sum(preds == y2.detach().cpu()).float())
    print(acc)

# torch.Size([1, 2, 5])
# torch.Size([5, 2])
# torch.Size([5])
# tensor([1, 1, 1, 0, 1])
# tensor(0.)



