# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os

from .wan_animate_14B import animate_14B
from .wan_i2v_A14B import i2v_A14B

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


WAN_CONFIGS = {
    'i2v-A14B': i2v_A14B,
    'animate-14B': animate_14B,
}

SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
    '704*1280': (704, 1280),
    '1280*704': (1280, 704),
    '1024*704': (1024, 704),
    '704*1024': (704, 1024),
}

MAX_AREA_CONFIGS = {
    '720*1280': 720 * 1280,
    '1280*720': 1280 * 720,
    '480*832': 480 * 832,
    '832*480': 832 * 480,
    '704*1280': 704 * 1280,
    '1280*704': 1280 * 704,
    '1024*704': 1024 * 704,
    '704*1024': 704 * 1024,
}

SUPPORTED_SIZES = {
    'i2v-A14B': ('720*1280', '1280*720', '480*832', '832*480'),
    'animate-14B': ('720*1280', '1280*720')
}
