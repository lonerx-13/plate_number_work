# 字符分割与模板匹配识别模块
from .segment_v3 import segment_with_fallback
from .segment_lpr import split_char
from .matcher import TemplateMatcher
from .recognizer import TemplateRecognizer, SegmentMethod

# HOG 特征匹配（使用 skimage）
try:
    from .deep_feature import HOGFeatureExtractor, HOGTemplateMatcher
    # 向后兼容：保留旧名称作为别名
    from .deep_feature import ResNetFeatureExtractor, DeepTemplateMatcher
    __all__ = ['segment_with_fallback', 'split_char',
               'TemplateMatcher', 'TemplateRecognizer', 'SegmentMethod',
               'HOGFeatureExtractor', 'HOGTemplateMatcher',
               'ResNetFeatureExtractor', 'DeepTemplateMatcher']  # 兼容旧代码
except ImportError:
    __all__ = ['segment_with_fallback', 'split_char',
               'TemplateMatcher', 'TemplateRecognizer', 'SegmentMethod']
