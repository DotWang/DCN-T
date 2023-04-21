import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.backbone import build_backbone
from einops import rearrange
from utils.gensp.ssn_sp import ssn_iter


def find_surrounding(input, l, h_shift_unit=1, w_shift_unit=1):
    # input should be padding as (c, 1+ n_h+1, 1+n_w+1)
    input_pd = F.pad(input.unsqueeze(0).unsqueeze(0).float(), (w_shift_unit,w_shift_unit,h_shift_unit,h_shift_unit), 'replicate')
    input_pd = input_pd.squeeze(0)

    # assign to ...
    top     = input_pd[:, :-2 * h_shift_unit,          w_shift_unit:-w_shift_unit]
    bottom  = input_pd[:, 2 * h_shift_unit:,           w_shift_unit:-w_shift_unit]
    left    = input_pd[:, h_shift_unit:-h_shift_unit,  :-2 * w_shift_unit]
    right   = input_pd[:, h_shift_unit:-h_shift_unit,  2 * w_shift_unit:]

    center = input_pd[:,h_shift_unit:-h_shift_unit,w_shift_unit:-w_shift_unit]

    bottom_right    = input_pd[:, 2 * h_shift_unit:,   2 * w_shift_unit:]
    bottom_left     = input_pd[:, 2 * h_shift_unit:,   :-2 * w_shift_unit]
    top_right       = input_pd[:, :-2 * h_shift_unit,  2 * w_shift_unit:]
    top_left        = input_pd[:, :-2 * h_shift_unit,  :-2 * w_shift_unit]

    center_surrounding = torch.cat((     top_left,    top,      top_right,
                                        left,        center,      right,
                                        bottom_left, bottom,    bottom_right), dim=0) # 9,n_h,n_w
    
    pixel_surrounding =  torch.repeat_interleave(
        torch.repeat_interleave(center_surrounding, l, dim=-1), l, dim=-2).unsqueeze(0) #1, 9, n_h*l, n_w*l
        
    return pixel_surrounding
    
    
def gen_sim_matrix(pixel_surrounding_idxs, c_features, p_features, n_iter):

    b, k, c = c_features.shape
    
    b, n, _ = pixel_surrounding_idxs.shape
    
    pixel_surrounding_idxs_repeat = pixel_surrounding_idxs.unsqueeze(-1).repeat(1,1,1,c).reshape(b,-1, c).type(torch.int64)  # b,n,9->b,n,9,c->b,n*9,c
    
    p_features = p_features.reshape(b,c,n).permute(0,2,1) # b, n, c
    
    for i in range(n_iter):
    
        surrounding_c_features = torch.gather(c_features, dim=1, index = pixel_surrounding_idxs_repeat).reshape(b,n,-1,c).permute(0,1,3,2) # b, n*9, c -> b, n, 9, c -> b, n, c, 9
    
        similarity = torch.matmul(p_features.unsqueeze(2), surrounding_c_features).squeeze(2) # b, n, 9
    
        similarity = F.softmax(similarity, dim=-1)
        
        M = torch.zeros(b, n, k).cuda()

        M = M.scatter_(2, pixel_surrounding_idxs, similarity).permute(0, 2, 1) # b, n, k -> b, k, n
        
        c_features = torch.bmm(M, p_features) / torch.sum(M, dim=-1, keepdim=True) # b, k, c
        
    return M, c_features
    

    # pixel_surrounding_idxs: b, n, 9 
    # c_feature: b, k, c


def gen_HR(p_features, K=16, n_iter=10):
    
    b, c, h, w = p_features.shape
    
    l = int(math.sqrt(h*w*1.0/K)) # size of one grid
    
    n_h = int(h/l)
    n_w = int(w/l)
    
    # center feature
    
    c_features = F.adaptive_avg_pool2d(p_features, (n_h, n_w)).reshape(b,c,-1).permute(0, 2, 1) # b,c,k -> b, k, c
    
    hr_idxs = torch.arange(0, n_h * n_w).reshape(n_h, n_w).cuda()
    
    pixel_surrounding_idxs = find_surrounding(hr_idxs, l)
    
    pixel_surrounding_idxs = pixel_surrounding_idxs.repeat(b,1,1,1) # b, 9, n_h*l, n_w*l
    
    if (n_h*l == h) and (n_w*l == w):
        pass
    else:
         pixel_surrounding_idxs = F.interpolate(pixel_surrounding_idxs, size=(h,w), mode='nearest')
         
    pixel_surrounding_idxs = pixel_surrounding_idxs.reshape(b, -1, h*w).permute(0,2,1).type(torch.int64) # b, 9, h, w -> b, 9, n -> b, n, 9
    
    # 迭代聚类，生成相似度矩阵
    
    M, c_features = gen_sim_matrix(pixel_surrounding_idxs, c_features, p_features, n_iter)
    
    _, HR = torch.max(M, dim=1)
    
    HR = HR.reshape(b, h, w).int()
    
    return M, HR, c_features

class Conv3x3BNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3x3BNReLU, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block1(x)
        return x

class DWConv1(nn.Module):
    def __init__(self, dim=1024):
        super(DWConv1, self).__init__()
        self.dwconv = nn.Conv1d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=1024):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class MHCA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):

        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots
        # attn = dots
        # vis_tmp(dots)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # vis_tmp2(out)

        return out

class MHSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)# chunk, split the tensor to a list
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)


        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, MHCA(dim, heads = heads,
                                                        dim_head = dim_head, dropout = dropout,
                                                        softmax=softmax))),
                Residual(PreNorm(dim, FFN(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, m, mask = None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)
            x = ff(x)
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, MHSA(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FFN(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class GlobalAggregation(nn.Module):

    def __init__(self, args, channel):

        super(GlobalAggregation, self).__init__()

        head = args.ga_head_num

        self.transformer_encoder = Transformer(dim=channel, depth=1, heads=head,
                                               dim_head=channel // head,
                                               mlp_dim=2 * channel, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=channel, depth=1,
                                                      heads=head, dim_head=channel // head, mlp_dim=2 * channel,
                                                      dropout=0,
                                                      softmax=True)

        self.pos = DWConv1(channel)

        self.channel = channel

    def forward(self, des, x):

        c,h,w = x[0].shape

        batch = len(des)

        gre = []

        for i in range(batch):

            di = des[i].unsqueeze(0) # 1, s, c

            xpi = self.pos(di.permute(0, 2, 1)).permute(0,2,1)

            xi = x[i].permute(1, 2, 0).reshape(-1, self.channel)

            encoder_qkv = di + xpi

            decoder_q = xi.unsqueeze(0) # 1,n,c

            decoder_kv = self.transformer_encoder(encoder_qkv)

            recons = self.transformer_decoder(decoder_q, decoder_kv)

            recons = recons.squeeze(0).permute(1,0).reshape(c, h, w)

            gre.append(recons)

        gre = torch.stack(gre,dim=0)

        return gre


class RegionAdaption(nn.Module):

    def __init__(self, args, channel):

        super(RegionAdaption, self).__init__()

        print(args)

        head = args.ra_head_num

        self.transformer_encoder = Transformer(dim=channel, depth=1, heads = head,
                                       dim_head=channel // head,
                                       mlp_dim=2*channel, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim = channel, depth=1,
                                                      heads=head, dim_head= channel // head, mlp_dim=2*channel,
                                                      dropout=0,
                                                      softmax=True)



    def forward(self, sp, x, xp):

        c, h, w = x.shape

        x = x.permute(1, 2, 0).reshape(-1, c)
        xp = xp.permute(1, 2, 0).reshape(-1, c)

        sp_values = torch.unique(sp)

        sp_num = sp_values.shape[0]

        x_r = torch.zeros(x.shape).cuda()

        d_r = torch.zeros([sp_num,c]).cuda() # mean value of region

        for i in range(sp_num):

            region_pos_2d = torch.nonzero(sp == int(sp_values[i]),as_tuple=False)

            region_pos_1d = region_pos_2d[:, 0] * w + region_pos_2d[:, 1]

            encoder_qkv = x[region_pos_1d] + xp[region_pos_1d]

            #decoder_q = x[region_pos_1d].unsqueeze(0)

            decoder_kv = self.transformer_encoder(encoder_qkv.unsqueeze(0))

            #recons = self.transformer_decoder(decoder_q, decoder_kv)

            x_r[region_pos_1d] = decoder_kv.squeeze(0)

            d_r[i] = torch.mean(x_r[region_pos_1d],dim=0)

        x_r = x_r.permute(1, 0).reshape(c, h, w)

        return x_r, d_r

import numpy as np

class posi_attn(nn.Module):

    def __init__(self, args, channel):

        super(posi_attn,self).__init__()

        self.args = args

        self.pos = DWConv(channel)

        self.ra = RegionAdaption(args, channel)
        self.ga = GlobalAggregation(args, channel)

    def forward(self, x):

        b, _, x_h, x_w = x.shape

        #np.save('F_feature.npy',x.squeeze(0).detach().cpu().numpy())

        xp1 = self.pos(x)

        _, sps, _ = ssn_iter(x, num_spixels=self.args.groups, n_iter=10)
        
        # _, sps, _  = gen_HR(x, K=self.args.groups, n_iter=10)

        sps = sps + 1

        sps = sps.reshape(b, x_h, x_w)

        r_features = []

        r_descriptors = []

        for i in range(b):
            r_feature, r_descriptor = self.ra(sps[i], x[i], xp1[i])

            r_features.append(r_feature)

            r_descriptors.append(r_descriptor)

        # r_features = torch.stack(r_features, dim=0) #b,c,h,w

        # r_descriptors = torch.stack(r_descriptors, dim=0) #b,s,c

        # xp2 = self.pos2(r_descriptors.permute(0,2,1))

        ra = torch.stack(r_features,dim=0)

        #np.save('F_rac.npy',ra.squeeze(0).detach().cpu().numpy())

        g_features = self.ga(r_descriptors, r_features)

        #np.save('F_gac.npy', g_features.squeeze(0).detach().cpu().numpy())

        return g_features

class chan_attn(nn.Module):

    def __init__(self, dim):
        super(chan_attn, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(dim)

        head = 4

        channel = dim*dim

        self.transformer_encoder = Transformer(dim=channel, depth=1, heads=head,
                                               dim_head=channel // head,
                                               mlp_dim=2 * channel, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=channel, depth=1,
                                                      heads=head, dim_head=channel // head, mlp_dim=2 * channel,
                                                      dropout=0,
                                                      softmax=True)

        self.pos = DWConv1(channel)

    def forward(self, x):

        b, _, x_h, x_w = x.shape

        x_pool = self.pool(x)

        b, _, h, w = x_pool.shape

        x_pool = x_pool.reshape(b,-1, h*w) # b,n,c

        xp = self.pos(x_pool.permute(0, 2, 1)).permute(0, 2, 1) # b,n,c

        encoder_qkv = x_pool + xp

        #decoder_q = x_pool

        recons = self.transformer_encoder(encoder_qkv)

        #recons = self.transformer_decoder(decoder_q, decoder_kv)

        recons = recons.reshape(b, -1, h, w)

        recons = F.interpolate(recons,size=(x_h, x_w),mode='bilinear',align_corners=False)

        return recons

class spec_attn(nn.Module):

    def __init__(self, channel, dim):

        super(spec_attn,self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(dim)

        head = 4

        channel = dim * dim * channel

        self.transformer_encoder = Transformer(dim=channel, depth=1, heads=head,
                                               dim_head=channel // head,
                                               mlp_dim=2 * channel, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=channel, depth=1,
                                                      heads=head, dim_head=channel // head, mlp_dim=2 * channel,
                                                      dropout=0,
                                                      softmax=True)
        self.pos = DWConv1(channel)

    def forward(self, x):

        b, _, x_h, x_w = x.shape

        x_pool = self.pool(x)

        b, _, h, w = x_pool.shape

        x_pool = x_pool.reshape(1,b,-1) #1,b,c

        xp = self.pos(x_pool.permute(0, 2, 1)).permute(0, 2, 1)  # 1,b,c

        encoder_qkv = x_pool + xp

        sre = self.transformer_encoder(encoder_qkv)

        sre = sre.squeeze(0).reshape(b,-1,1,1)

        return torch.sigmoid(sre)

class rat_model(nn.Module):

    def __init__(self, args, classes, in_channels, freeze_bn=False, freeze_backbone=False):

        super(rat_model, self).__init__()

        self.args = args

        self.backbone_name = args.backbone

        self.backbone = build_backbone(args,self.backbone_name,in_channels)

        if 'hrnet' in self.backbone_name:

            self.conv = Conv3x3BNReLU(int(sum(self.backbone.channels)),self.backbone.channels[-1] // 2)

        else:
            self.conv = Conv3x3BNReLU(self.backbone.channels[-1], self.backbone.channels[-1] // 2)


        self.pa = posi_attn(args, self.backbone.channels[-1] // 2)

        self.ca = chan_attn(64)

        #self.ba = spec_attn(self.backbone.channels[-1] // 2, 1)


        self.conv_dsn = nn.Sequential(
            Conv3x3BNReLU(self.backbone.channels[-2], self.backbone.channels[-2] // 2),
            nn.Conv2d(self.backbone.channels[-2] // 2, classes, kernel_size=1, bias=False)
        )

        self.conv_out = nn.Sequential(
            Conv3x3BNReLU(self.backbone.channels[-1], self.backbone.channels[-1] // 2),
            nn.Conv2d(self.backbone.channels[-1] // 2, classes, kernel_size=1, bias=False)
        )

        #self.freeze_backbone()

        self._initialize_weights()

    def forward(self, x):

        b, c, h, w = x.shape

        x = self.backbone(x)

        if self.training:

            x_dsn = self.conv_dsn(x[-2])
            x_dsn = F.interpolate(x_dsn, size=(h, w), mode='bilinear', align_corners=False)

        if 'hrnet' in self.backbone_name:

            x_h, x_w = x[0].size(2), x[0].size(3)

            x_2 = F.interpolate(x[1], size=(x_h, x_w), mode='bilinear', align_corners=False)
            x_3 = F.interpolate(x[2], size=(x_h, x_w), mode='bilinear', align_corners=False)
            x_4 = F.interpolate(x[3], size=(x_h, x_w), mode='bilinear', align_corners=False)

            x = torch.cat([x[0], x_2, x_3, x_4], 1)  # n,c,h,w

        else:

            x = x[-1]

        _, _, x_h, x_w = x.shape

        x = self.conv(x)

        x_pa = self.pa(x)
        #
        #x_ca = self.ca(x)
        #
        #x_ba = self.ba(x)

        #x_a = x_ca #+ x_ca#)*x_ba

        #features = torch.cat([x, x],1)

        x = self.conv_out(torch.cat([x_pa,x],1))

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        if self.training:
            return x, x_dsn
        else:
            return x

    def _initialize_weights(self):
        modules = [self.conv, self.pa, self.ca, self.conv_dsn, self.conv_out]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d):
                    nn.init.xavier_normal_(m[1].weight.data, gain=1)
                    if m[1].bias is not None:
                        nn.init.constant_(m[1].bias.data, 0)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_backbone(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for p in m[1].parameters():
                    p.requires_grad = False

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                # if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                #         or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.conv, self.pa, self.ca, self.conv_dsn, self.conv_out]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                # if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                #         or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
