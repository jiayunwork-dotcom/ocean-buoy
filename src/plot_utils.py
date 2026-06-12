import matplotlib
import matplotlib.pyplot as plt
import platform
import os
from matplotlib import font_manager

CN_FONT_PRIORITY = [
    'PingFang SC', 'Hiragino Sans GB', 'Heiti SC',
    'Microsoft YaHei', 'SimHei', 'STHeiti',
    'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
    'Noto Sans CJK SC', 'Noto Sans SC',
    'Source Han Sans CN', 'Source Han Sans SC',
    'Arial Unicode MS'
]

_MAC_FONT_PATHS = [
    '/System/Library/Fonts/PingFang.ttc',
    '/System/Library/Fonts/STHeiti Light.ttc',
    '/System/Library/Fonts/STHeiti Medium.ttc',
    '/System/Library/Fonts/Hiragino Sans GB.ttc',
    '/Library/Fonts/Arial Unicode.ttf',
    '/Library/Fonts/Songti.ttc',
]

_WIN_FONT_PATHS = [
    'C:\\Windows\\Fonts\\msyh.ttc',
    'C:\\Windows\\Fonts\\msyh.ttf',
    'C:\\Windows\\Fonts\\msyhbd.ttc',
    'C:\\Windows\\Fonts\\simhei.ttf',
    'C:\\Windows\\Fonts\\simsun.ttc',
]


def _rebuild_font_cache():
    try:
        font_manager._load_fontmanager(try_read_cache=False)
    except Exception:
        pass


def setup_chinese_font():
    _rebuild_font_cache()
    system = platform.system()
    font_name = None
    font_path = None
    for font in CN_FONT_PRIORITY:
        try:
            fp = font_manager.findfont(font, fallback_to_default=False)
            if fp and os.path.exists(fp):
                font_name = font
                font_path = fp
                break
        except Exception:
            continue
    if font_path is None:
        search_paths = []
        if system == 'Darwin':
            search_paths = _MAC_FONT_PATHS
        elif system == 'Windows':
            search_paths = _WIN_FONT_PATHS
        elif system == 'Linux':
            search_paths = [
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
            ]
        for fp in search_paths:
            if os.path.exists(fp):
                try:
                    font_manager.fontManager.addfont(fp)
                    prop = font_manager.FontProperties(fname=fp)
                    font_name = prop.get_name()
                    font_path = fp
                    break
                except Exception:
                    continue
    if font_name:
        matplotlib.rcParams['font.sans-serif'] = [font_name] + [
            f for f in matplotlib.rcParams.get('font.sans-serif', []) if f != font_name
        ]
        matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False
    return font_name


def get_available_chinese_fonts():
    available = []
    for font in CN_FONT_PRIORITY:
        try:
            fp = font_manager.findfont(font, fallback_to_default=False)
            if fp and os.path.exists(fp):
                available.append(font)
        except Exception:
            continue
    return available
