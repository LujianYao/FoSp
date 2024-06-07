# Copyright (c) OpenMMLab. All rights reserved.
import collections
from copy import copy
from copy import deepcopy

from mmcv.utils import build_from_cfg

from ..builder import PIPELINES


@PIPELINES.register_module()
class InpaintingCompose(object):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        # if type(data['img']) == list:
        #     print(data)
        # if len(self.transforms) == 3:
        #     print(self.transforms)
        data_inpainting = deepcopy(data)
        data_inpainting['img_prefix'] = data['inpainting_prefix']
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        
        for t in self.transforms:
            data_inpainting = t(data_inpainting)
            if data is None:
                return None
        
        # data['diff_img'] = data['img'].data - data_inpainting['img'].data
        # if type(data['img']) == list:
        #     for idx in range(len(data['img'])):
        #         data['img'][idx] = [data['img'][idx], data['img'][idx].data - data_inpainting['img'][idx].data]
        # else:
        #     data['img'] = [data['img'], data['img'].data - data_inpainting['img'].data]
        data['img'] = [data['img'], data_inpainting['img']]

        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
