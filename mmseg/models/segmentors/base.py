# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.distributed as dist
import copy
from mmcv.runner import BaseModule, auto_fp16
import cv2


class BaseSegmentor(BaseModule, metaclass=ABCMeta):
    """Base class for segmentors."""

    def __init__(self, init_cfg=None):
        super(BaseSegmentor, self).__init__(init_cfg)
        self.fp16_enabled = False

    @property
    def with_neck(self):
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_auxiliary_head(self):
        """bool: whether the segmentor has auxiliary head"""
        return hasattr(self,
                       'auxiliary_head') and self.auxiliary_head is not None

    @property
    def with_decode_head(self):
        """bool: whether the segmentor has decode head"""
        return hasattr(self, 'decode_head') and self.decode_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        """Placeholder for extract features from images."""
        pass

    @abstractmethod
    def encode_decode(self, img, img_metas):
        """Placeholder for encode images with backbone and decode into a
        semantic segmentation map of the same size as input."""
        pass

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """Placeholder for Forward function for training."""
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        """Placeholder for single image test."""
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Placeholder for augmentation test."""
        pass

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got '
                                f'{type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) != '
                             f'num of image meta ({len(img_metas)})')
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        for img_meta in img_metas:
            ori_shapes = [_['ori_shape'] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_['img_shape'] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_['pad_shape'] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']))

        return outputs

    def val_step(self, data_batch, optimizer=None, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        log_vars_ = dict()
        for loss_name, loss_value in log_vars.items():
            k = loss_name + '_val'
            log_vars_[k] = loss_value

        outputs = dict(
            loss=loss,
            log_vars=log_vars_,
            num_samples=len(data_batch['img_metas']))

        return outputs

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, raise assertion error
        # to prevent GPUs from infinite waiting.
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()) + '\n')
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
    
    def overlay_mask(self, image, mask, alpha=0.8):
        temp_mask = (mask != 0).astype(np.float32)
        temp_mask = np.expand_dims(temp_mask[:, :, 0], axis=2)
        over_image = image * (1 - alpha) + mask * alpha
        overlay = over_image * temp_mask + image * (1 - temp_mask)
        # overlay = image.copy()
        # overlay[mask == color] = alpha * mask[mask == color] + (1 - alpha) * image[mask == color]
        return overlay

    def show_result(self,
                    img,
                    result,
                    palette=None,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    opacity=0.5):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        gt_path = '/workspace/ylj/SmokeSeg/data/SmokeSeg10K/SegmentationClass/' + out_file.split('/')[-1].replace('.jpg', '.png')
        gt_seg_map = mmcv.imread(gt_path)
        save_focus_map = False
        if len(result) == 3:
            save_focus_map = True
        img = mmcv.imread(img)
        img = img.copy()
        if len(result) == 2:
            seg = result[0][0]
        elif save_focus_map:
            seg = result[0][0]
            focus_map = result[2][0][0]
            focus_map = (1 - focus_map).astype(np.float32)
            # focus_map_32 = cv2.resize(focus_map, (32, 32))
            # focus_map_32[focus_map_32>0]=1
            # focus_map_512 = cv2.resize(focus_map_32, (512, 512))
            # focus_map_512[focus_map_512 > 0] = 1
            # focus_map = focus_map_512.copy()


        else:
            seg = result[0]
        if palette is None:
            if self.PALETTE is None:
                # Get random state before set seed,
                # and restore random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(
                    0, 255, size=(len(self.CLASSES), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE
        palette = np.array(palette)
        assert palette.shape[0] == len(self.CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        if save_focus_map:
            color_seg_focus = np.zeros((focus_map.shape[0], focus_map.shape[1], 3), dtype=np.uint8)
        # palette[1] = [128, 0, 0]
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        if save_focus_map:
            palette_focus_map = copy.deepcopy(palette)
            palette_focus_map[1] = [0, 0, 255] # 设定颜色
            for label, color in enumerate(palette_focus_map):
                color_seg_focus[focus_map == label, :] = color
            # convert to BGR
            color_seg_focus = color_seg_focus[..., ::-1]

        origin_img = copy.deepcopy(img)
        # img = self.overlay_mask(origin_img, color_seg, 0.3)
        # img = origin_img * (1 - opacity) + color_seg * opacity # 这个背景会黑一些
        img = img + color_seg * opacity # 
        # gt_img = self.overlay_mask(origin_img, gt_seg_map, 0.3)
        # gt_img = origin_img * (1 - opacity) + gt_seg_map * opacity
        gt_img = origin_img + gt_seg_map * opacity

        img[img>255] = 255 
        gt_img[gt_img>255] = 255 
        img = img.astype(np.uint8)
        gt_img = gt_img.astype(np.uint8)

        if save_focus_map:
            # focus_image = origin_img * (1 - opacity) + color_seg_focus * opacity

            # focus_image = img + color_seg_focus * opacity 
            focus_image = self.overlay_mask(origin_img, color_seg_focus, 0.3)
            focus_gt_image = self.overlay_mask(gt_img, color_seg_focus, 0.3) # 这个是正宗的颜色覆盖，背景不会黑，里面要是img，就是在上面叠加的基础上再叠加

            focus_gt_image[focus_gt_image>255] = 255 
            focus_image[focus_image>255] = 255 

            focus_gt_image = focus_gt_image.astype(np.uint8)
            focus_image = focus_image.astype(np.uint8)



        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
            mmcv.imwrite(color_seg, out_file.replace('.jpg', '_pred.png'))
            if save_focus_map:
                mmcv.imwrite(focus_image, out_file.replace('.jpg', '_focus.jpg'))
                mmcv.imwrite(focus_gt_image, out_file.replace('.jpg', '_focus+gt+img.jpg'))
                mmcv.imwrite(gt_img, out_file.replace('.jpg', '_gt+img.jpg'))



        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img
