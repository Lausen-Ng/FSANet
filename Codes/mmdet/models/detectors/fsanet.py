# -----------------------------------------------------------------------------
# Licensed under the MIT License.
# The code is based on MMDetection (https://github.com/open-mmlab/mmdetection).
# @Author  : Jixiang Wu
# @Time    : 2022/5/21 下午4:42
# -----------------------------------------------------------------------------
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class FSANet(SingleStageDetector):
    """Implementation of `FSANet <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(FSANet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)
