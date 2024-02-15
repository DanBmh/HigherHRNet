# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import transforms as T


FLIP_CONFIG = {
    'COCO': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
    ],
    'COCO_WITH_CENTER': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17
    ],
    'COCO_EXTRA_KEYPOINTS': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17, 18
    ],
    'COCO_WHOLE_BODY': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15,  # body
        20, 21, 22, 17, 18, 19,  # foot
        39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23,  # face1
        49, 48, 47, 46, 45, 44, 43, 42, 41, 40,  # face2
        50, 51, 52, 53,  # face3
        58, 57, 56, 55, 54,  # face4
        68, 67, 66, 65, 62, 61, 60, 59,  # face5
        70, 69, 64, 63,  # face6
        77, 76, 75, 74, 73, 72, 71,  # face7
        82, 81, 80, 79, 78, # face8
        88, 87, 86, 85, 84, 83,  # face9
        90, 89, # face10
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,  # hand1
        91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111  # hand2
    ],
    'CROWDPOSE': [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13
    ],
    'CROWDPOSE_WITH_CENTER': [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14
    ]
}


def build_transforms(cfg, is_train=True):
    assert is_train is True, 'Please only use build_transforms for training.'
    assert isinstance(cfg.DATASET.OUTPUT_SIZE, (list, tuple)), 'DATASET.OUTPUT_SIZE should be list or tuple'
    if is_train:
        max_rotation = cfg.DATASET.MAX_ROTATION
        min_scale = cfg.DATASET.MIN_SCALE
        max_scale = cfg.DATASET.MAX_SCALE
        max_translate = cfg.DATASET.MAX_TRANSLATE
        input_size = cfg.DATASET.INPUT_SIZE
        output_size = cfg.DATASET.OUTPUT_SIZE
        flip = cfg.DATASET.FLIP
        scale_type = cfg.DATASET.SCALE_TYPE
    else:
        scale_type = cfg.DATASET.SCALE_TYPE
        max_rotation = 0
        min_scale = 1
        max_scale = 1
        max_translate = 0
        input_size = 512
        output_size = [128]
        flip = 0

    # coco_flip_index = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    # if cfg.DATASET.WITH_CENTER:
        # coco_flip_index.append(17)
    if 'coco' in cfg.DATASET.DATASET:
        dataset_name = 'COCO'
    elif 'crowd_pose' in cfg.DATASET.DATASET:
        dataset_name = 'CROWDPOSE'
    else:
        raise ValueError('Please implement flip_index for new dataset: %s.' % cfg.DATASET.DATASET)
    if cfg.DATASET.WITH_CENTER:
        coco_flip_index = FLIP_CONFIG[dataset_name + '_WITH_CENTER']
    elif cfg.DATASET.ADD_EXTRA_JOINTS:
        coco_flip_index = FLIP_CONFIG[dataset_name + '_EXTRA_KEYPOINTS']
    elif cfg.DATASET.USE_WHOLE_BODY:
        coco_flip_index = FLIP_CONFIG[dataset_name + '_WHOLE_BODY']
    else:
        coco_flip_index = FLIP_CONFIG[dataset_name]

    transforms = T.Compose(
        [
            T.RandomAffineTransform(
                input_size,
                output_size,
                max_rotation,
                min_scale,
                max_scale,
                scale_type,
                max_translate,
                scale_aware_sigma=cfg.DATASET.SCALE_AWARE_SIGMA
            ),
            T.RandomHorizontalFlip(coco_flip_index, output_size, flip),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    return transforms
