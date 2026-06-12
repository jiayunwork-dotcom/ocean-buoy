import matplotlib
import matplotlib.pyplot as plt
import platform
from matplotlib import font_manager

CN_FONT_PRIORITY = [
    'PingFang SC', 'Hiragino Sans GB', 'Heiti SC',
    'Microsoft YaHei', 'SimHei', 'STHeiti',
    'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
    'Noto Sans CJK SC', 'Noto Sans SC',
    'Source Han Sans CN', 'Source Han Sans SC'
]


def setup_chinese_font():
    system = platform.system()
    font_name = None
    for font in CN_FONT_PRIORITY:
        try:
            font_manager.findfont(font, fallback_to_default=False)
            font_name = font
            break
        except Exception:
            continue
    if font_name is None:
        if system == 'Darwin':
            font_path = '/System/Library/Fonts/PingFang.ttc'
            try:
                font_manager.fontManager.addfont(font_path)
                font_name = 'PingFang SC'
            except Exception:
                pass
        elif system == 'Windows':
            font_paths = [
                'C:\\Windows\\Fonts\\msyh.ttc',
                'C:\\Windows\\Fonts\\simhei.ttf',
            ]
            for fp in font_paths:
                try:
                    font_manager.fontManager.addfont(fp)
                    font_name = font_manager.FontProperties(fname=fp).get_name()
                    break
                except Exception:
                    continue
    if font_name:
        matplotlib.rcParams['font.sans-serif'] = [font_name] + matplotlib.rcParams['font.sans-serif']
        matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False
    return font_name


def get_available_chinese_fonts():
    available = []
    for font in CN_FONT_PRIORITY:
        try:
            font_manager.findfont(font, fallback_to_default=False)
            available.append(font)
        except Exception:
            continue
    return available
