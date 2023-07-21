# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from ..builder import BACKBONES, build_backbone


class SToD(BaseModule):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        x = rearrange(x, 'b c (h h2) (w w2) -> b (h2 w2 c) h w', h2=self.bs, w2=self.bs)
        return x
    

class DToS(BaseModule):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        x = rearrange(x, 'b (h2 w2 c) h w -> b c (h h2) (w w2)', h2=self.bs, w2=self.bs)
        return x


class ca(BaseModule):
    def __init__(self, in_planes, ratio=4, conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(ca, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=True),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(self.avg_pool(x))
        out = self.sigmoid(out)*x
        # print(out.shape)
        return out


class sa(BaseModule):
    def __init__(self, kernel_size=7, scale=2, conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(sa, self).__init__()

        self.q = ConvModule(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size//2,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg)
        self.k = nn.Sequential(
                    ConvModule(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size//2,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),
                    nn.AvgPool2d((scale+1, scale+1), padding=(scale+1)//2, stride=scale)
                    )
        self.v = nn.Sequential(
                    ConvModule(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size//2,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),
                    nn.AvgPool2d((scale+1, scale+1), padding=(scale+1)//2, stride=scale)
                    )

        self.w = nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b,_,h,w = x.shape
        feat=x
        x = torch.mean(x, dim=1, keepdim=True)
        q = self.q(x).view(b, 1, -1)
        q = q.permute(0, 2, 1)
        k = self.k(x).view(b, 1, -1)
        v = self.v(x).view(b, 1, -1)
        v = v.permute(0, 2, 1)
        sim = torch.matmul(q, k)
        sim = F.softmax(sim, dim=-1)
        x = torch.matmul(sim, v)
        x = x.permute(0,2,1).contiguous()
        x = x.view(b, 1, h, w)
        x = self.sigmoid(self.w(x))*feat
        return x


class contextPath(BaseModule):
    def __init__(self, backbone_cfg, 
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(contextPath, self).__init__()
        self.resnet = build_backbone(backbone_cfg)

    def forward(self, x):
        feat4, feat8, feat16, feat32 = self.resnet(x)

        return feat4, feat8, feat16, feat32


class spatialPath(nn.Module):
    def __init__(self, conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(spatialPath, self).__init__()
        self.s1 = nn.Sequential(
                SToD(8),
                ConvModule(
                        in_channels=192,
                        out_channels=128,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'))
                )
        
        self.s2 = ConvModule(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'))

        self.kernel = nn.Sequential(
                    ConvModule(
                        in_channels=128*2,
                        out_channels=128*2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=128*2,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='Sigmoid')),
                        
                    ConvModule(
                        in_channels=128*2,
                        out_channels=128*2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=128*2,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='Sigmoid')),

                    ConvModule(
                        in_channels=128*2,
                        out_channels=128*2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='Sigmoid')),
                        
                    )


    def forward(self, x):
        s1 = self.s1(x)
        
        fr = torch.fft.rfftn(s1, dim=tuple(range(2, x.ndim)))
        cen_fr = torch.fft.fftshift(fr, dim=tuple(range(2, x.ndim)))
        k = torch.view_as_real(cen_fr)
        k = rearrange(k, 'b c h w z -> b (c z) h w', z=2)
        k = self.kernel(k)
        k = rearrange(k, 'b (c z) h w -> b c h w z', z=2).contiguous()
        k = torch.view_as_complex(k)
        cen_fr = cen_fr*k

        fr = torch.fft.ifftshift(cen_fr, dim=tuple(range(2, x.ndim)))
        output = torch.fft.irfftn(fr, dim=tuple(range(2, x.ndim))).abs()
        
        output = self.s2(output)
        
        return output

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)



class FSA(nn.Module):
    def __init__(self, ch_8, ch_16, ch_32, ch_sp, out_ch, num_feat=7,conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super(FSA, self).__init__()
        
        self.num_feat = num_feat
        self.out_ch = out_ch
        
        self.p0 = ConvModule(
                        in_channels=ch_32,
                        out_channels=out_ch,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'))
        self.p1 = nn.Sequential(
                    nn.AvgPool2d(17, 8, padding=8),
                    ConvModule(
                        in_channels=ch_32,
                        out_channels=out_ch,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'))                   
                    )
        self.p2 = nn.Sequential(
                   nn.AvgPool2d(9, 4, padding=4),
                   ConvModule(
                        in_channels=ch_32,
                        out_channels=out_ch,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'))  
                    )
        self.p3 = nn.Sequential(
                    nn.AvgPool2d(5, 2, padding=2),
                    ConvModule(
                        in_channels=ch_32,
                        out_channels=out_ch,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'))  
                    )
        self.p4 = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    ConvModule(
                        in_channels=ch_32,
                        out_channels=out_ch,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'))  
                    )
        
        self.s16 = ConvModule(
                        in_channels=ch_16,
                        out_channels=out_ch,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'))  

        self.sp = ConvModule(
                        in_channels=ch_sp,
                        out_channels=out_ch,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'))  
        
        self.c128to1 = nn.Sequential(
                        ConvModule(
                                    in_channels=out_ch*num_feat,
                                    out_channels=num_feat,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=num_feat,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg,
                                    act_cfg=dict(type='ReLU'))  ,
                
                        nn.Conv2d(num_feat, num_feat, 1, bias=True),
                        nn.ReLU(True),
                        nn.Conv2d(num_feat, num_feat, 1, bias=True),
                        nn.Sigmoid()
                        )
        
        self.ca = ca(out_ch, norm_cfg=norm_cfg)
        self.sa = sa(norm_cfg=norm_cfg)

        self.init_weight()

    def forward(self, feat8, feat16, feat32, featSp):
        p0 = self.p0(feat32)
        p1 = self.p1(feat32)
        p2 = self.p2(feat32)
        p3 = self.p3(feat32)
        p4 = self.p4(feat32)
        
        s16 = self.s16(feat16)
        sp = self.sp(featSp)
        
        hw = sp.shape[2:]
        p0 = F.interpolate(p0, size=hw, mode='bilinear', align_corners=False)
        p1 = F.interpolate(p1, size=hw, mode='bilinear', align_corners=False)
        p2 = F.interpolate(p2, size=hw, mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, size=hw, mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=hw, mode='bilinear', align_corners=False)
        s16 = F.interpolate(s16, size=hw, mode='bilinear', align_corners=False)
        
        fAll = torch.cat([p0,p1,p2,p3,p4,s16, sp], 1)
        spAttn = self.c128to1(fAll)
        spAttn = repeat(spAttn, 'b c h w -> b (c g) h w', g=self.out_ch)
        fAll = fAll*spAttn
        fAll = rearrange(fAll, 'b (g c) h w -> b g c h w', c=self.out_ch)
        fAll = self.sa(self.ca(torch.sum(fAll, 1)))

        return fAll

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


@BACKBONES.register_module()
class FDLNet(BaseModule):

    def __init__(self,
                 backbone_cfg,
                 in_channels=3,
                 backbone_channels=(128, 256, 512),
                 spatialPath_channels=128,
                 out_indices=(0, 1, 2),
                 out_channels=128,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):

        super(FDLNet, self).__init__(init_cfg=init_cfg)

        self.out_indices = out_indices
        self.cp = contextPath(backbone_cfg, norm_cfg=norm_cfg)
        self.sp = spatialPath(norm_cfg=norm_cfg)
        self.fsa = FSA(backbone_channels[0], 
                       backbone_channels[1], 
                       backbone_channels[2], 
                       spatialPath_channels, 
                       out_channels,
                       norm_cfg=norm_cfg)


    def forward(self, x):
        feat4, feat8, feat16, feat32= self.cp(x)
        feat_sp = self.sp(x)
        feat_fuse = self.fsa(feat8, feat16, feat32, feat_sp)

        outs = [feat_fuse, feat_sp, feat8, feat16]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)
