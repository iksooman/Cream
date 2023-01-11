import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module.Linear_super import LinearSuper
from model.module.layernorm_super import LayerNormSuper
from model.module.multihead_super import AttentionSuper
from model.module.embedding_super import PatchembedSuper
from model.module.mbconv_super import DynamicMBConvLayer
from model.utils import trunc_normal_
from model.utils import DropPath
import numpy as np


def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Vision_TransformerSuper(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., pre_norm=True, scale=False, gp=False, relative_position=False, change_qkv=False, abs_pos = True, max_relative_position=14,
                 early_conv=False):
        super(Vision_TransformerSuper, self).__init__()
        # the configs of super arch
        self.super_embed_dim = embed_dim
        # self.super_embed_dim = args.embed_dim
        self.super_mlp_ratio = mlp_ratio
        self.super_layer_num = depth
        self.super_num_heads = num_heads
        self.super_dropout = drop_rate
        self.super_attn_dropout = attn_drop_rate
        self.num_classes = num_classes
        self.pre_norm=pre_norm
        self.scale=scale
        self.in_chans=in_chans

        self.ops_econv    = ['conv', 'conv', 'conv', 'conv']
        self.out_ch_econv = [24, 48, 96, 192]
        self.stride_econv = [2, 2, 2, 1]
        self.kernel_econv = [3, 3, 3, 3]

        def create_earlyconv(prev_out, out_ch_econv, stride_econv, kernel_econv):
            econv_list = []
            for e_ops, e_out, e_st, e_kernel in zip(self.ops_econv, self.out_ch_econv, self.stride_econv, kernel_econv):
                if e_ops == 'conv':
                    for ops_econv in [nn.Conv2d(prev_out, e_out, e_kernel, stride=e_st, padding=(e_kernel-1)//2, bias=False, groups=1), nn.BatchNorm2d(e_out), nn.ReLU()]:
                        econv_list.append(ops_econv)
                elif e_ops == 'max':
                    for ops_econv in [nn.Conv2d(prev_out, e_out, 1, stride=1, padding=0, bias=False, groups=1), nn.BatchNorm2d(e_out), nn.ReLU(), nn.MaxPool2d(2, 2)]:
                        econv_list.append(ops_econv)
                prev_out = e_out
            return nn.Sequential(*econv_list)

        if early_conv:
            self.early_conv = create_earlyconv(self.in_chans, self.out_ch_econv, self.stride_econv, self.kernel_econv)
            patch_in_channel = self.out_ch_econv[-1]
            img_down = 1
            for each_stride in self.stride_econv:
                img_down *= each_stride

        else:
            self.early_conv = nn.Identity()
            patch_in_channel = in_chans
            img_down = 1


        self.patch_embed_super = PatchembedSuper(img_size=img_size//img_down, patch_size=patch_size//img_down,
                                                 in_chans=patch_in_channel, embed_dim=embed_dim)
        self.gp = gp

        # configs for the sampled subTransformer
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_layer_num = None
        self.sample_num_heads = None
        self.sample_dropout = None
        self.sample_output_dim = None

        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        for i in range(depth):
            self.blocks.append(TransformerEncoderLayer(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                                       qkv_bias=qkv_bias, qk_scale=qk_scale, dropout=drop_rate,
                                                       attn_drop=attn_drop_rate, drop_path=dpr[i],
                                                       pre_norm=pre_norm, scale=self.scale,
                                                       change_qkv=change_qkv, relative_position=relative_position,
                                                       max_relative_position=max_relative_position, add_conv=True if i < 0 else False))

        # parameters for vision transformer
        num_patches = self.patch_embed_super.num_patches

        self.abs_pos = abs_pos
        if self.abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)

        # self.pos_drop = nn.Dropout(p=drop_rate)
        if self.pre_norm:
            self.norm = LayerNormSuper(super_embed_dim=embed_dim)


        # classifier head
        self.head = LinearSuper(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'rel_pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def set_sample_config(self, config: dict):
        self.sample_embed_dim = config['embed_dim']
        self.sample_mlp_ratio = config['mlp_ratio']
        self.sample_layer_num = config['layer_num']
        self.sample_num_heads = config['num_heads']
        self.sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[0], self.super_embed_dim)
        self.patch_embed_super.set_sample_config(self.sample_embed_dim[0])
        self.sample_output_dim = [out_dim for out_dim in self.sample_embed_dim[1:]] + [self.sample_embed_dim[-1]]
        for i, blocks in enumerate(self.blocks):
            # not exceed sample layer number
            if i < self.sample_layer_num:
                sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[i], self.super_embed_dim)
                sample_attn_dropout = calc_dropout(self.super_attn_dropout, self.sample_embed_dim[i], self.super_embed_dim)
                blocks.set_sample_config(is_identity_layer=False,
                                        sample_embed_dim=self.sample_embed_dim[i],
                                        sample_mlp_ratio=self.sample_mlp_ratio[i],
                                        sample_num_heads=self.sample_num_heads[i],
                                        sample_dropout=sample_dropout,
                                        sample_out_dim=self.sample_output_dim[i],
                                        sample_attn_dropout=sample_attn_dropout,
                                        )
            # exceeds sample layer number
            else:
                blocks.set_sample_config(is_identity_layer=True)
        if self.pre_norm:
            self.norm.set_sample_config(self.sample_embed_dim[-1])
        self.head.set_sample_config(self.sample_embed_dim[-1], self.num_classes)

    def get_sampled_params_numel(self, config):
        numels = []

        # early conv params
        params = self.in_chans * self.out_ch_econv[0] * self.kernel_econv[0]
        for i in range(len(self.out_ch_econv)-1):
            params += self.out_ch_econv[i] * self.out_ch_econv[i+1] * self.kernel_econv[i] * self.kernel_econv[i]
        #early conv batch params
        for j in range(len(self.out_ch_econv)):
            params += self.out_ch_econv[j] * 4
        numels.append(params)

        self.set_sample_config(config)
        for name, module in self.named_modules():
            if hasattr(module, 'calc_sampled_param_num'):
                if name.split('.')[0] == 'blocks' and int(name.split('.')[1]) >= config['layer_num']:
                    continue
                numels.append(module.calc_sampled_param_num())

        return sum(numels) + self.sample_embed_dim[0]* (2 +self.patch_embed_super.num_patches)

    def get_alphas(self, sigmoid=True):
        return_list = []
        for blocks in self.blocks:
            return_list.append(blocks.get_alpha(sigmoid=sigmoid))

        return return_list

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.patch_embed_super.get_complexity(sequence_length)
        total_flops += np.prod(self.pos_embed[..., :self.sample_embed_dim[0]].size()) / 2.0
        for blk in self.blocks:
            total_flops +=  blk.get_complexity(sequence_length+1)
        total_flops += self.head.get_complexity(sequence_length+1)
        return total_flops
    def forward_features(self, x):
        B = x.shape[0]
        x = self.early_conv(x)
        x = self.patch_embed_super(x)
        cls_tokens = self.cls_token[..., :self.sample_embed_dim[0]].expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.abs_pos:
            x = x + self.pos_embed[..., :self.sample_embed_dim[0]]

        x = F.dropout(x, p=self.sample_dropout, training=self.training)

        # start_time = time.time()
        for blk in self.blocks:
            x = blk(x)
        # print(time.time()-start_time)
        if self.pre_norm:
            x = self.norm(x)

        if self.gp:
            return torch.mean(x[:, 1:] , dim=1)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class CustomizedMBConvLayer(nn.Module):

    def __init__(self, channel, kernel_size, expand_ratio, class_emb=True):
        super().__init__()
        self.mbconv = DynamicMBConvLayer(in_channel_list=channel, out_channel_list=channel,
                                         kernel_size_list=kernel_size, expand_ratio_list=expand_ratio,
                                         use_se=True)
        self.class_emb = class_emb

    def forward(self, x):
        residual = x

        B, S, C = x.shape
        H = int(math.sqrt(S))
        if self.class_emb:
            x = x[:, 1:]
        x = x.transpose(1, 2)
        x = torch.reshape(x, (B, C, H, H))
        x = self.mbconv(x)
        x = torch.reshape(x, (B, C, H*H))
        x = x.transpose(1, 2)

        if self.class_emb:
            return torch.cat((x[:, :1, :]*0., x), dim=1) + residual
        else:
            return x + residual


class LinearEvaluation(nn.Module):

    def __init__(self, node_num):
        super().__init__()
        '''
        self.fc1 = LinearSuper(super_in_dim=node_num, super_out_dim=node_num*2)
        self.fc2 = LinearSuper(super_in_dim=node_num*2, super_out_dim=node_num*2)
        self.fc3 = LinearSuper(super_in_dim=node_num*2, super_out_dim=node_num)
        '''
        self.fc1 = nn.Sequential(
                        nn.Linear(node_num, max(128, node_num*4), bias=False),
                        nn.BatchNorm1d(max(128, node_num*2)),
                        nn.ReLU()
                        )
        self.fc2 = nn.Sequential(
                        nn.Linear(max(128, node_num*4), max(128, node_num*4), bias=False),
                        nn.BatchNorm1d(max(128, node_num*2)),
                        nn.ReLU()
                        )
        self.fc3 = nn.Sequential(
                        nn.Linear(max(128, node_num*4), node_num, bias=False),
                        # nn.BatchNorm1d(node_num),
                        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class MaxLayer(nn.Module):
    def __init__(self, dims=0):
        super().__init__()
        self.dims = dims

    def forward(self, x, dim=None):
        input_max, max_indices = torch.max(x, dim if dim is not None else self.dims)
        return input_max


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments which
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, dropout=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, pre_norm=True, scale=False,
                 relative_position=False, change_qkv=False, max_relative_position=14, add_conv=False):
        super().__init__()

        # the configs of super arch of the encoder, three dimension [embed_dim, mlp_ratio, and num_heads]
        self.super_embed_dim = dim
        self.super_mlp_ratio = mlp_ratio
        self.super_ffn_embed_dim_this_layer = int(mlp_ratio * dim)
        self.super_num_heads = num_heads
        self.normalize_before = pre_norm
        self.super_dropout = attn_drop
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale = scale
        self.relative_position = relative_position
        # self.super_activation_dropout = getattr(args, 'activation_dropout', 0)

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_num_heads_this_layer = None
        self.sample_scale = None
        self.sample_dropout = None
        self.sample_attn_dropout = None

        self.add_conv = True
        if self.add_conv:
            print('Mixed Layer')
            self.conv = CustomizedMBConvLayer(channel=dim, kernel_size=3, expand_ratio=6, class_emb=True)
            self.alpha = torch.nn.parameter.Parameter(nn.init.constant_(torch.Tensor(1), 0.), requires_grad=True)
            self.beta  = torch.nn.parameter.Parameter(nn.init.constant_(torch.Tensor(1), 0.), requires_grad=True)

        self.is_identity_layer = None
        self.attn = AttentionSuper(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=dropout, scale=self.scale, relative_position=self.relative_position, change_qkv=change_qkv,
            max_relative_position=max_relative_position
        )

        self.attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.ffn_layer_norm = LayerNormSuper(self.super_embed_dim)
        # self.dropout = dropout
        self.activation_fn = gelu
        # self.normalize_before = args.encoder_normalize_before

        self.fc1 = LinearSuper(super_in_dim=self.super_embed_dim, super_out_dim=self.super_ffn_embed_dim_this_layer)
        self.fc2 = LinearSuper(super_in_dim=self.super_ffn_embed_dim_this_layer, super_out_dim=self.super_embed_dim)


    def set_sample_config(self, is_identity_layer, sample_embed_dim=None, sample_mlp_ratio=None, sample_num_heads=None, sample_dropout=None, sample_attn_dropout=None, sample_out_dim=None):

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.sample_embed_dim = sample_embed_dim
        self.sample_out_dim = sample_out_dim
        self.sample_mlp_ratio = sample_mlp_ratio
        self.sample_ffn_embed_dim_this_layer = int(sample_embed_dim*sample_mlp_ratio)
        self.sample_num_heads_this_layer = sample_num_heads

        self.sample_dropout = sample_dropout
        self.sample_attn_dropout = sample_attn_dropout
        self.attn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

        self.attn.set_sample_config(sample_q_embed_dim=self.sample_num_heads_this_layer*64, sample_num_heads=self.sample_num_heads_this_layer, sample_in_embed_dim=self.sample_embed_dim)

        self.fc1.set_sample_config(sample_in_dim=self.sample_embed_dim, sample_out_dim=self.sample_ffn_embed_dim_this_layer)
        self.fc2.set_sample_config(sample_in_dim=self.sample_ffn_embed_dim_this_layer, sample_out_dim=self.sample_out_dim)

        self.ffn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

        if self.add_conv:
            self.conv.mbconv.set_subnet_config(in_channel=self.sample_embed_dim)

    def roundup(self, num):
        return int(num) + 1 if (num - int(num)) >= 0.5 else int(num)

    def get_alpha(self, sigmoid):
        if sigmoid:
            return torch.sigmoid(self.alpha)
        else:
            return self.alpha

    def forward(self, x):
        """
        Args:
            x (Tensor): input to the layer of shape `(batch, patch_num , sample_embed_dim)`

        Returns:
            encoded output of shape `(batch, patch_num, sample_embed_dim)`
        """
        if self.is_identity_layer:
            return x

        # compute attn
        # start_time = time.time()

        sample_prob = torch.nn.functional.softmax(torch.cat([self.alpha, self.beta]))
        if self.training:
            sample_choice = [torch.bernoulli(sample_prob[0])]
        else:
            sample_choice = [self.roundup(sample_prob[0])]
            print(sample_choice)

        if sample_choice[0] > 0.5:
            x_conv = self.conv(x)
            b = 1. - sample_prob[0].detach()
            c = sample_prob[0] + b
            return x_conv * c

        else:
            residual = x
            x = self.maybe_layer_norm(self.attn_layer_norm, x, before=True)
            x = self.attn(x)
            x = F.dropout(x, p=self.sample_attn_dropout, training=self.training)
            x = self.drop_path(x)
            x = residual + x
            x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)
            # print("attn :", time.time() - start_time)
            # compute the ffn
            # start_time = time.time()
            residual = x
            x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
            x = self.activation_fn(self.fc1(x))
            x = F.dropout(x, p=self.sample_dropout, training=self.training)
            x = self.fc2(x)
            x = F.dropout(x, p=self.sample_dropout, training=self.training)
            if self.scale:
                x = x * (self.super_mlp_ratio / self.sample_mlp_ratio)
            x = self.drop_path(x)
            x = residual + x
            x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)
            # print("ffn :", time.time() - start_time)

            b = sample_prob[1].detach()
            c = 1. - sample_prob[1] + b
            return x * c

        '''
        if self.add_conv:
            a = torch.sigmoid(self.alpha)
            return a*x  + (1.-a)*self.conv(x)
        else:
            return x
        '''

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x
    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.is_identity_layer:
            return total_flops
        total_flops += self.attn_layer_norm.get_complexity(sequence_length+1)
        total_flops += self.attn.get_complexity(sequence_length+1)
        total_flops += self.ffn_layer_norm.get_complexity(sequence_length+1)
        total_flops += self.fc1.get_complexity(sequence_length+1)
        total_flops += self.fc2.get_complexity(sequence_length+1)
        return total_flops

def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim





