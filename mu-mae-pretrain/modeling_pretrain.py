import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial

from modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_



def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'pretrain_videomae_small_patch16_224',
    'pretrain_videomae_base_patch16_224', 
    'pretrain_videomae_large_patch16_224', 
    'pretrain_videomae_huge_patch16_224',
]


class SensorPatchEmbed(nn.Module):
    """ Sensor to Patch Embedding
    """
    def __init__(self, in_chans=24,  embed_dim=384):
        super().__init__()
        self.num_patches = 16
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=1, stride=1)

    def forward(self, x, **kwargs):
        # FIXME look at relaxing size constraints
        x = self.proj(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2, use_checkpoint=False,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint
        self.patch_embed_phone = SensorPatchEmbed(24, embed_dim) # 24=128*3/16
        self.patch_embed_watch = SensorPatchEmbed(24, embed_dim)
        self.patch_embed_gyro = SensorPatchEmbed(24, embed_dim)
        self.patch_embed_orientation = SensorPatchEmbed(24, embed_dim)


        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
            self.acc_phone_pos_embed = get_sinusoid_encoding_table(16, embed_dim) #16 is patch number of sensor
            self.acc_watch_pos_embed = get_sinusoid_encoding_table(16, embed_dim)
            self.gyro_pos_embed = get_sinusoid_encoding_table(16, embed_dim)
            self.ori_pos_embed = get_sinusoid_encoding_table(16, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask,acc_phone_data,acc_watch_data,gyro_data,orientation_data,acc_phone_mask,acc_watch_mask,gyro_mask,ori_mask):
        #video
        _, _, T, _, _ = x.shape
        # print("x before patch embedding in modeling encoder function: " + str(x.shape))
        # x before patch embedding in modeling encoder function: torch.Size([32, 3, 16, 224, 224])

        x = self.patch_embed(x)
        # print("x patch embedding in modeling encoder function: " + str(x.shape))
        # x patch embedding in modeling encoder function: torch.Size([32, 1568, 384])

        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        # print("self.pos_embed shape in modeling encoder function: " + str(self.pos_embed.shape))
        # print("x after add pos embed in modeling encoder function: " + str(x.shape))
        # self.pos_embed shape in modeling encoder function: torch.Size([1, 1568, 384])
        # x after add pos embed in modeling encoder function: torch.Size([32, 1568, 384])
        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible
        # print("x_vis shape in modeling encoder function: " + str(x_vis.shape))
        # x_vis shape in modeling encoder function: torch.Size([32, 160, 384])

        # sensor phone
        # print("acc_phone_data before patch embedding in modeling encoder function: " + str(acc_phone_data.shape))
        phone_x = self.patch_embed_phone(acc_phone_data)
        # print("phone_x after patch embeding in modeling encoder function: " + str(phone_x.shape))
        phone_x = phone_x + self.acc_phone_pos_embed.type_as(phone_x).to(phone_x.device).clone().detach()
        # print("self.acc_phone_pos_embed shape in modeling encoder function: " + str(self.acc_phone_pos_embed.shape))
        # print("phone_x after add pos embed in modeling encoder function: " + str(phone_x.shape))
        phone_x_vis = phone_x[~acc_phone_mask].reshape(B, -1, C) # ~mask means visible
        # print("phone_x_vis shape in modeling encoder function: " + str(phone_x_vis.shape))
        # sensor watch
        # print("acc_watch_data before patch embedding in modeling encoder function: " + str(acc_watch_data.shape))
        watch_x = self.patch_embed_watch(acc_watch_data)
        watch_x = watch_x + self.acc_watch_pos_embed.type_as(watch_x).to(watch_x.device).clone().detach()
        # print("self.acc_watch_pos_embed shape in modeling encoder function: " + str(self.acc_watch_pos_embed.shape))
        # print("watch_x after add pos embed in modeling encoder function: " + str(watch_x.shape))
        watch_x_vis = watch_x[~acc_watch_mask].reshape(B, -1, C) # ~mask means visible
        # print("watch_x_vis shape in modeling encoder function: " + str(watch_x_vis.shape))
        # sensor gyro
        # print("gyro_data before patch embedding in modeling encoder function: " + str(gyro_data.shape))
        gyro_x = self.patch_embed_gyro(gyro_data)
        gyro_x = gyro_x + self.gyro_pos_embed.type_as(gyro_x).to(gyro_x.device).clone().detach()
        # print("self.gyro_pos_embed shape in modeling encoder function: " + str(self.gyro_pos_embed.shape))
        # print("gyro_x after add pos embed in modeling encoder function: " + str(gyro_x.shape))
        gyro_x_vis = gyro_x[~gyro_mask].reshape(B, -1, C) # ~mask means visible
        # print("gyro_x_vis shape in modeling encoder function: " + str(gyro_x_vis.shape))
        # sensor ori
        # print("orientation_data before patch embedding in modeling encoder function: " + str(orientation_data.shape))
        ori_x = self.patch_embed_orientation(orientation_data)
        ori_x = ori_x + self.ori_pos_embed.type_as(ori_x).to(ori_x.device).clone().detach()
        # print("self.ori_pos_embed shape in modeling encoder function: " + str(self.ori_pos_embed.shape))
        # print("ori_x after add pos embed in modeling encoder function: " + str(ori_x.shape))
        ori_x_vis = ori_x[~ori_mask].reshape(B, -1, C) # ~mask means visible
        # print("ori_x_vis shape in modeling encoder function: " + str(ori_x_vis.shape))



        x_combine = torch.cat((x_vis, phone_x_vis, watch_x_vis,gyro_x_vis,ori_x_vis), dim=1)

        if self.use_checkpoint:
            for blk in self.blocks:
                x_combine = checkpoint.checkpoint(blk, x_combine)
        else:   
            for blk in self.blocks:
                x_combine = blk(x_combine)

        x_combine = self.norm(x_combine)
        # print("x_combine shape before return in modeling encoder function: " + str(x_combine.shape))
        # x_vis shape before return in modeling encoder function: torch.Size([32, 160, 384])
        return x_combine

    def forward(self, x, mask,acc_phone_data,acc_watch_data,gyro_data,orientation_data,acc_phone_mask,acc_watch_mask,gyro_mask,ori_mask):
        x = self.forward_features(x, mask,acc_phone_data,acc_watch_data,gyro_data,orientation_data,acc_phone_mask,acc_watch_mask,gyro_mask,ori_mask)
        # print("x shape after forward in modeling encoder function: " + str(x.shape))
        # x shape after forward in modeling encoder function: torch.Size([32, 160, 384])

        x = self.head(x)
        # print("x shape after head in modeling encoder function: " + str(x.shape))
        # x shape after head in modeling encoder function: torch.Size([32, 160, 384])
        return x

class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=None, num_patches=196, tubelet_size=2, use_checkpoint=False
                 ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * tubelet_size * patch_size ** 2 
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:   
            for blk in self.blocks:
                x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x

class PretrainSensorTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,  num_classes=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=None, num_patches=196,  use_checkpoint=False
                 ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.use_checkpoint = use_checkpoint
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        # print("number class " + str(num_classes))
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))
        # print("decoder return shape:" + str(x.shape))

        return x


class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=1536, #  decoder_num_classes=768,
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 use_checkpoint=False,
                 tubelet_size=2,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 ):
        super().__init__()
        self.sensor_decoder_number_class = 24
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_learnable_pos_emb=use_learnable_pos_emb)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint)
        self.phone_decoder = PretrainSensorTransformerDecoder(
            num_patches=self.encoder.patch_embed_phone.num_patches,
            num_classes=self.sensor_decoder_number_class,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_checkpoint=use_checkpoint)
        self.watch_decoder = PretrainSensorTransformerDecoder(
            num_patches=self.encoder.patch_embed_watch.num_patches,
            num_classes=self.sensor_decoder_number_class,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_checkpoint=use_checkpoint)
        self.gyro_decoder = PretrainSensorTransformerDecoder(
            num_patches=self.encoder.patch_embed_gyro.num_patches,
            num_classes=self.sensor_decoder_number_class,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_checkpoint=use_checkpoint)
        self.ori_decoder = PretrainSensorTransformerDecoder(
            num_patches=self.encoder.patch_embed_orientation.num_patches,
            num_classes=self.sensor_decoder_number_class,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_checkpoint=use_checkpoint)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.vid_pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)
        self.phone_pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed_phone.num_patches, decoder_embed_dim)
        self.watch_pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed_watch.num_patches, decoder_embed_dim)
        self.gyro_pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed_gyro.num_patches, decoder_embed_dim)
        self.ori_pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed_orientation.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def get_decoder_result(self, x_data, x_encoder_data, x_mask, pos_embed, decoder_model):
        # x_vis shape after encoder_to_decoder in modeling function: torch.Size([32, 160, 192])
        B, N, C = x_encoder_data.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        # print("pos_embed shape before expand in modeling function: " + str(pos_embed.shape))
        # vid_pos_embed shape before expand in modeling function: torch.Size([1, 1568, 192])
        expand_pos_embed = pos_embed.expand(B, -1, -1).type_as(x_data).to(x_data.device).clone().detach()
        # print("expand_pos_embed shape after expand in modeling function: " + str(expand_pos_embed.shape))
        # expand_pos_embed shape after expand in modeling function: torch.Size([32, 1568, 192])
        pos_emd_vis = expand_pos_embed[~x_mask].reshape(B, -1, C)
        # print("pos_emd_vis shape after expand in modeling function: " + str(pos_emd_vis.shape))
        # pos_emd_vis shape after expand in modeling function: torch.Size([32, 160, 192])
        pos_emd_mask = expand_pos_embed[x_mask].reshape(B, -1, C)
        # print("pos_emd_mask shape after expand in modeling function: " + str(pos_emd_mask.shape))
        # print("self.mask_token shape after expand in modeling function: " + str(self.mask_token.shape))
        # pos_emd_mask shape after expand in modeling function: torch.Size([32, 1408, 192])
        # self.mask_token shape after expand in modeling function: torch.Size([1, 1, 192])
        x_full = torch.cat([x_encoder_data + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        # print("x_full shape before decoder in modeling function: " + str(x_full.shape))
        # x_full shape before decoder in modeling function: torch.Size([32, 1568, 192])
        decoder_x = decoder_model(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]
        # print("decoder_x shape after decoder in modeling function: " + str(decoder_x.shape))
        # x shape after decoder in modeling function: torch.Size([32, 1408, 1536])
        return decoder_x

    def forward(self, x, mask,acc_phone_data,acc_watch_data,gyro_data,orientation_data,acc_phone_mask,acc_watch_mask,gyro_mask,ori_mask):

        # encoder
        B, _, T, _, _ = x.shape
        x_all = self.encoder(x, mask,acc_phone_data,acc_watch_data,gyro_data,orientation_data,acc_phone_mask,acc_watch_mask,gyro_mask,ori_mask) # [B, N_vis, C_e]
        x_all = self.encoder_to_decoder(x_all) # [B, N_vis, C_d]
        # print("x_all shape after encoder_to_decoder in modeling function: " + str(x_all.shape))
        vid_vis_patch_num = torch.sum(~mask)/B
        phone_vis_patch_num = torch.sum(~acc_phone_mask)/B
        watch_vis_patch_num = torch.sum(~acc_watch_mask)/B
        gyro_vis_patch_num = torch.sum(~gyro_mask)/B
        ori_vis_patch_num = torch.sum(~ori_mask)/B
        sensor_split_size = [int(vid_vis_patch_num.item()),int(phone_vis_patch_num.item()),int(watch_vis_patch_num.item()),int(gyro_vis_patch_num.item()),int(ori_vis_patch_num.item())]
        # print("sensor_split_size before encoder in modeling function: " + str(sensor_split_size))
        # get encoder result for each sensor
        sensor_split_tensors = torch.split(x_all, sensor_split_size, dim=1)
        x_vid_vis = sensor_split_tensors[0]
        x_phone_vis = sensor_split_tensors[1]
        x_watch_vis = sensor_split_tensors[2]
        x_gyro_vis = sensor_split_tensors[3]
        x_ori_vis = sensor_split_tensors[4]
        # print("x_vid_vis shape after encoder_to_decoder in modeling function: " + str(x_vid_vis.shape))
        # print("x_phone_vis shape after encoder_to_decoder in modeling function: " + str(x_phone_vis.shape))
        # print("x_watch_vis shape after encoder_to_decoder in modeling function: " + str(x_watch_vis.shape))
        # print("x_gyro_vis shape after encoder_to_decoder in modeling function: " + str(x_gyro_vis.shape))
        # print("x_ori_vis shape after encoder_to_decoder in modeling function: " + str(x_ori_vis.shape))

        # video data
        vid_x = self.get_decoder_result(x,x_vid_vis,mask,self.vid_pos_embed,self.decoder)
        # phone data
        phone_x = self.get_decoder_result(acc_phone_data, x_phone_vis, acc_phone_mask, self.phone_pos_embed,self.phone_decoder)
        # watch data
        watch_x = self.get_decoder_result(acc_watch_data, x_watch_vis, acc_watch_mask, self.watch_pos_embed,self.watch_decoder)
        # # gyro data
        gyro_x = self.get_decoder_result(gyro_data, x_gyro_vis, gyro_mask, self.gyro_pos_embed,self.gyro_decoder)
        # # ori data
        ori_x = self.get_decoder_result(orientation_data, x_ori_vis, ori_mask, self.ori_pos_embed,self.ori_decoder)

        return vid_x,phone_x,watch_x,gyro_x,ori_x

@register_model
def pretrain_videomae_small_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=1536, 
        decoder_embed_dim=192, 
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_videomae_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
 
@register_model
def pretrain_videomae_large_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=1024, 
        encoder_depth=24, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1536, 
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_videomae_huge_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=1280, 
        encoder_depth=32, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1536, 
        decoder_embed_dim=640,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
