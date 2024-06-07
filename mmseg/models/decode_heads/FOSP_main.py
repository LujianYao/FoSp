######## FoSp ########
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.base_decode_head import BaseDecodeHead
from mmseg.ops import resize
import numpy as np
from inpainter_setting import dnnlib
import PIL.Image
from inpainter_setting.networks.inpainter import Generator
from torchvision.transforms import transforms 

@HEADS.register_module()
class FOSP(BaseDecodeHead):
    def __init__(self, interpolate_mode='bilinear', focus_threshold=0.85, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        self.focus_threshold = focus_threshold
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.foreground_channel_convs = nn.ModuleList()
        for i in range(num_inputs):
            in_channels = 512
            if i == 3:
                in_channels = 256
            self.foreground_channel_convs.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=self.in_channels[num_inputs - 1 - i],
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,)
                    )

        self.fusion_foreground_inputs_conv = nn.ModuleList()
        for i in range(num_inputs):
            self.fusion_foreground_inputs_conv.append(
                ConvModule(
                    in_channels=self.in_channels[num_inputs - 1 - i] * 2 + self.channels * (i),
                    out_channels=self.channels * (i + 1),
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_final_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.cls_seg_1_4 = nn.Conv2d(self.in_channels[0], 1, kernel_size=1)
        self.cls_seg_1_8 = nn.Conv2d(self.in_channels[1] * 2, 1, kernel_size=1)
        self.cls_seg_1_16 = nn.Conv2d(self.in_channels[2] * 2, 1, kernel_size=1)
        self.cls_seg_1_32 = nn.Conv2d(self.in_channels[3], 1, kernel_size=1)
        self.fusion_bothway_mask = nn.Conv2d(2, 1, kernel_size=1)
        self.conv_1_4_8 = nn.Conv2d(self.in_channels[0], self.in_channels[1], kernel_size=1)
        self.conv_1_32_16 = nn.Conv2d(self.in_channels[3], self.in_channels[2], kernel_size=1)


        
        # init inpainter
        # The inpainting model can be replaced with different models as needed. We are referencing MAT (CVPR22) in this case.
        # https://github.com/fenglinglwb/MAT
        resolution = 512
        device = torch.device('cuda')
        net_res = 512 if resolution > 512 else resolution
        self.G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3).to(device).eval().requires_grad_(False)


    def generate_images(self, Generater, image, mask, z, label, noise_mode):
            output = Generater(image, mask, z, label, truncation_psi=1, noise_mode=noise_mode)
            return output

    def get_querymap(self, logits):
        logits = logits[:, 0, :, :].unsqueeze(1).mul(255).add_(0.5).clamp_(0, 255)
        logits[logits < 255] = 0
        logits = logits / 255
        return logits

    def save_image(self, image, path):
        image = (image[0].unsqueeze(0).permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
        image = image[0].cpu().numpy()
        image = image[:,:,::-1]
        if image.shape[2] == 1:
            image = image[:, :, 0]
        PIL.Image.fromarray(image).save(path)

    
    def BCG(self, inputs):
        #### Bidirectional Casacde Generator ####
        out_1_4 = self.cls_seg_1_4(inputs[0]) 
        out_1_4_querymap = torch.sigmoid(out_1_4) 
        out_1_4_add_query =  torch.mul(inputs[0], out_1_4_querymap) + inputs[0] # add inputs
        query_1_4 = transforms.Resize(inputs[1].size()[2:])(self.conv_1_4_8(out_1_4_add_query)) # resize
        out_1_8 = self.cls_seg_1_8(torch.cat([query_1_4, inputs[1]], dim=1)) # query cat origin
        out_1_32 = self.cls_seg_1_32(inputs[3])
        out_1_32_querymap = torch.sigmoid(out_1_32) 
        out_1_32_add_query = torch.mul(inputs[3], out_1_32_querymap) + inputs[3] # add inputs
        query_1_32 = transforms.Resize(inputs[2].size()[2:])(self.conv_1_32_16(out_1_32_add_query))
        out_1_16 = self.cls_seg_1_16(torch.cat([query_1_32, inputs[2]], dim=1)) # query cat origin
        focus_map_logits = self.fusion_bothway_mask(torch.cat([transforms.Resize(out_1_16.size()[2:])(out_1_8), out_1_16], dim=1))
        focus_map = torch.sigmoid(focus_map_logits)
        focus_map = transforms.Resize([512, 512])(focus_map)
        return focus_map, focus_map_logits, out_1_4, out_1_8, out_1_16, out_1_32

    def add_query(self, input_feature, query_map):
        query_map = transforms.Resize(input_feature.size()[2:])(query_map)
        out_feature = torch.mul(input_feature, query_map) + input_feature
        return out_feature

    def get_query_map(self, logits):
        query_map = torch.sigmoid(logits)
        return query_map


    def _transform_Seg(self, seg, transformed_size):
        seg_size = seg.shape[-1]
        num_tile = int(seg_size / transformed_size)
        seg = seg.squeeze(1)
        seg_arrays = np.array(seg.cpu().detach())
        tile_batch = []
        for seg_array in seg_arrays:
            bytelength = seg_array.nbytes // seg_array.size
            tiled = np.lib.stride_tricks.as_strided(seg_array, 
                                            shape=(transformed_size, transformed_size, num_tile, num_tile),
                                            strides=(seg_size * num_tile * bytelength, num_tile * bytelength, seg_size * bytelength, bytelength), 
                                            writeable=False)
            new_tiled = tiled.copy()
            new_tiled[new_tiled > 0] = 1
            new_tiled = new_tiled.sum(axis=2).sum(axis=2)
            new_tiled[new_tiled > 0] = 1
            new_tiled = np.expand_dims(new_tiled, axis=0)
            tile_batch.append(new_tiled)
        transformed_seg_array = np.concatenate(tile_batch, axis=0)
        transformed_seg = torch.from_numpy(transformed_seg_array).unsqueeze(1).cuda()
        return transformed_seg

    def grid_mask(self, mask):
        origin_shape = mask.shape[2:]
        mask_32 = resize(input=mask,
                      size=[32, 32],
                      mode=self.interpolate_mode,
                      align_corners=self.align_corners)
        mask_32[mask_32!=1] = 0
        mask_512 = resize(input=mask_32,
                      size=origin_shape,
                      mode=self.interpolate_mode,
                      align_corners=self.align_corners)
        mask_512[mask_512!=1] = 0
        mask_512 = torch.as_tensor(mask_512)
        return mask_512

    def forward(self, inputs, origin_img):
        num_inputs = len(self.in_channels)
        inputs = self._transform_inputs(inputs)

        ###### Focus: Bidirectional Cascade Generator (BCG) ######
        focus_map, focus_map_logits, out_1_4, out_1_8, out_1_16, out_1_32 = self.BCG(inputs)
        focus_map_mask_re = 1 - focus_map # reverse focus_map_mask
        focus_map_mask = focus_map_mask_re.clone()
        focus_map_mask[focus_map_mask < self.focus_threshold] = 0 
        focus_map_mask[focus_map_mask >= self.focus_threshold] = 1 
        focus_map_mask = focus_map_mask.to(torch.int64)

        ####### Separation #######
        device = torch.device('cuda')
        label = torch.zeros([focus_map_mask.shape[0], self.G.c_dim], device=device)
        z = torch.from_numpy(np.random.randn(focus_map_mask.shape[0], self.G.z_dim)).to(device)
        origin_img_ = origin_img.clone()
        if origin_img_.shape[2:] != [512, 512]:
            origin_img = resize(
                        input=origin_img,
                        size=[512, 512],
                        mode=self.interpolate_mode,
                        align_corners=self.align_corners)
        inpainting_feature = self.generate_images(self.G, origin_img, focus_map_mask.to(torch.float32), z, label, 'none')
        origin_feature = self.generate_images(self.G, origin_img, torch.ones_like(focus_map_mask).to(torch.float32), z, label, 'none')
        foreground = [abs(inpainting_feature[i] - origin_feature[i]) * 10 for i in range(len(inpainting_feature))]

        # Registration channel
        for i in range(len(foreground)):
            foreground[i] = self.foreground_channel_convs[i](foreground[i])

        # Registration scale
        if origin_img_.shape[2:] != [512, 512]:
            for idx in range(len(inputs)):
                foreground[idx] = resize(
                        input=foreground[idx],
                        size=inputs[num_inputs - 1 - idx].shape[2:],
                        mode=self.interpolate_mode,
                        align_corners=self.align_corners)

        ####### Domain Fusion ########
        for idx in range(len(inputs)):
            inputs[num_inputs - 1 - idx] = self.add_query(inputs[num_inputs - 1 - idx], focus_map)  
            x = torch.cat([inputs[num_inputs - 1 - idx], foreground[idx]], dim=1)
            conv = self.fusion_foreground_inputs_conv[idx]
            if idx == 0:
                temp_feature_map = conv(x)
                temp_feature_map = resize(
                        input=temp_feature_map,
                        size=inputs[num_inputs - 1 - idx - 1].shape[2:],
                        mode=self.interpolate_mode,
                        align_corners=self.align_corners)
            elif idx < 3:
                x = torch.cat([x, temp_feature_map], dim=1)
                temp_feature_map = conv(x)
                temp_feature_map = resize(
                        input=temp_feature_map,
                        size=inputs[num_inputs - 1 - idx - 1].shape[2:],
                        mode=self.interpolate_mode,
                        align_corners=self.align_corners)
            else:
                x = torch.cat([x, temp_feature_map], dim=1)
                temp_feature_map = conv(x)

        out = self.fusion_final_conv(temp_feature_map)
        out = self.cls_fusion_seg(out)
        torch.cuda.empty_cache()
        return out, [out_1_4, out_1_8, out_1_16, out_1_32, focus_map_logits]
