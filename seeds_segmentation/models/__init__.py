"""
=================================
Models for Semantic Segmentation
=================================
"""
from .baseline_fcn import fcn_baseline
from .baseline_unet import unet_baseline
from .baseline_res_unet import res_unet_baseline
from .baseline_segnet import segnet_baseline
from .mobilenetv2 import mobilenetv2
from .mobilenetv2_fcn import fcn_mobilenetv2
from .mobilenetv2_unet import unet_mobilenetv2
from .mobilenetv2_res_unet import res_unet_mobilenetv2
from .vgg16_unet import unet_vgg16