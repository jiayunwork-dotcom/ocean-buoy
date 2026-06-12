import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Tuple, Optional, List
import matplotlib.colors as mcolors

from .plot_utils import setup_chinese_font
setup_chinese_font()

from .quality_control import (
    QC_UNCHECKED, QC_GOOD, QC_SUSPECT, QC_ERROR, QC_MISSING, QC_INTERPOLATED, PARAMETERS
)

PARAM_NAMES_CN = {
    'wind_speed': '风速', 'wind_dir': '风向', 'air_temp': '气温',
    'pressure': '气压', 'Hs': '有效波高', 'Tz': '平均波周期',
    'wave_dir': '波向', 'SST': '海表温度', 'salinity': '盐度',
    'current_speed': '流速', 'current_dir': '流向'
}

QC_LEVEL_NAMES = {
    1: '范围检查', 2: '时间一致性', 3: '内部一致性',
    4: '气候学检查', 5: '尖峰检测', 6: '卡值检测', 7: '空间一致性'
}

QC_COLORS = {
    QC_UNCHECKED: 'gray',
    QC_GOOD: 'green',
    QC_SUSPECT: 'orange',
    QC_ERROR: 'red',
    QC_MISSING: 'purple',
    QC_INTERPOLATED: 'blue'
}

QC_LABELS = {
    QC_UNCHECKED: '未检',
    QC_GOOD: '合格',
    QC_SUSPECT: '可疑',
    QC_ERROR: '错误',
    QC_MISSING: '缺测',
    QC_INTERPOLATED: '插补'
}


def plot_qc_timeseries(df: pd.DataFrame, qc_codes: pd.DataFrame, buoy_id: str,
                       param: str, manual_overrides: Dict = None) -> Figure:
    buoy_df = df[df['buoy_id'] == buoy_id].sort_values('time')
    buoy_qc = qc_codes[qc_codes['buoy_id'] == buoy_id].sort_values('time')
    times = buoy_df['time'].values
    values = buoy_df[param].values
    codes = buoy_qc[param].values.copy()
    if manual_overrides:
        for i, t in enumerate(times):
            key = (buoy_id, pd.Timestamp(t), param)
            if key in manual_overrides:
                codes[i] = manual_overrides[key][0]
    fig, ax = plt.subplots(figsize=(14, 5))
    mask_valid = ~np.isnan(values)
    if np.any(mask_valid):
        ax.plot(times[mask_valid], values[mask_valid], 'b-', linewidth=0.8, alpha=0.5, label='原始值')
    for code, color in QC_COLORS.items():
        if code == QC_MISSING:
            continue
        mask = (codes == code) & ~np.isnan(values)
        if np.any(mask):
            ax.scatter(times[mask], values[mask], c=color, s=20, zorder=5,
                       label=QC_LABELS[code], alpha=0.8, edgecolors='none')
    param_names = {
        'wind_speed': '风速 (m/s)', 'wind_dir': '风向 (度)', 'air_temp': '气温 (℃)',
        'pressure': '气压 (hPa)', 'Hs': '有效波高 (m)', 'Tz': '平均波周期 (s)',
        'wave_dir': '波向 (度)', 'SST': '海表温度 (℃)', 'salinity': '盐度 (PSU)',
        'current_speed': '流速 (m/s)', 'current_dir': '流向 (度)'
    }
    ax.set_xlabel('时间')
    ax.set_ylabel(param_names.get(param, param))
    ax.set_title(f'{buoy_id} - {param_names.get(param, param)} 质控结果')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_qc_statistics(qc_codes: pd.DataFrame) -> Figure:
    param_counts = {}
    for param in PARAMETERS:
        if param in qc_codes.columns:
            counts = qc_codes[param].value_counts().to_dict()
            param_counts[param] = counts
    params = list(param_counts.keys())
    codes = [QC_GOOD, QC_SUSPECT, QC_ERROR, QC_MISSING, QC_INTERPOLATED]
    code_labels = [QC_LABELS[c] for c in codes]
    n_params = len(params)
    fig, axes = plt.subplots(1, min(4, n_params), figsize=(16, 4))
    if n_params == 1:
        axes = [axes]
    for i, param in enumerate(params[:4]):
        ax = axes[i]
        counts = param_counts[param]
        sizes = [counts.get(c, 0) for c in codes]
        colors = [QC_COLORS[c] for c in codes]
        total = sum(sizes)
        if total > 0:
            wedges, texts, autotexts = ax.pie(sizes, labels=code_labels, colors=colors,
                                               autopct='%1.1f%%', startangle=90)
            for t in autotexts:
                t.set_fontsize(8)
            for t in texts:
                t.set_fontsize(8)
        ax.set_title(param, fontsize=10)
    plt.suptitle('各参数质控码分布', fontsize=14)
    plt.tight_layout()
    return fig


def plot_qc_level_stats(level_stats: Dict) -> Figure:
    levels = sorted(level_stats.keys())
    fig, axes = plt.subplots(len(levels), 1, figsize=(12, 3 * len(levels)))
    if len(levels) == 1:
        axes = [axes]
    param_names_short = ['风速', '风向', '气温', '气压', 'Hs', 'Tz', '波向', 'SST', '盐度', '流速', '流向']
    for i, level in enumerate(levels):
        ax = axes[i]
        stats = level_stats[level]
        params = [p for p in PARAMETERS if p in stats]
        x = np.arange(len(params))
        width = 0.15
        codes = [QC_GOOD, QC_SUSPECT, QC_ERROR, QC_MISSING]
        for j, code in enumerate(codes):
            counts = [stats[p].get(code, 0) for p in params]
            ax.bar(x + j * width, counts, width, label=QC_LABELS[code], color=QC_COLORS[code], alpha=0.8)
        level_names = {1: '范围检查', 2: '时间一致性', 3: '内部一致性', 4: '气候学检查',
                       5: '尖峰检测', 6: '卡值检测', 7: '空间一致性'}
        ax.set_ylabel('数量')
        ax.set_title(f'第{level}级 - {level_names.get(level, str(level))}')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(param_names_short[:len(params)], rotation=45, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig


def apply_manual_override(qc_result, buoy_id: str, time: pd.Timestamp,
                          param: str, new_code: int, reason: str):
    key = (buoy_id, time, param)
    qc_result.manual_overrides[key] = (new_code, reason)
    return qc_result


def apply_batch_override(qc_result, buoy_id: str, start_time: pd.Timestamp,
                         end_time: pd.Timestamp, param: str, new_code: int, reason: str):
    for idx in range(len(qc_result.qc_codes)):
        row = qc_result.qc_codes.iloc[idx]
        if (row['buoy_id'] == buoy_id and
            start_time <= row['time'] <= end_time):
            key = (buoy_id, row['time'], param)
            qc_result.manual_overrides[key] = (new_code, reason)
    return qc_result


def extract_error_events(df: pd.DataFrame, qc_result) -> pd.DataFrame:
    events = []
    qc_codes = qc_result.qc_codes
    level_marks = qc_result.level_marks
    manual_overrides = qc_result.manual_overrides
    
    for idx in range(len(qc_codes)):
        row = qc_codes.iloc[idx]
        buoy_id = row['buoy_id']
        time = row['time']
        df_row = df[(df['buoy_id'] == buoy_id) & (df['time'] == time)]
        if len(df_row) == 0:
            continue
        df_row = df_row.iloc[0]
        for param in PARAMETERS:
            if param not in qc_codes.columns:
                continue
            final_code = qc_result.get_final_code(buoy_id, time, param)
            if final_code != QC_ERROR:
                continue
            mark_level = qc_result.get_mark_level(buoy_id, time, param)
            original_value = df_row[param] if param in df_row.index else None
            events.append({
                'time': time,
                'buoy_id': buoy_id,
                'param': param,
                'param_name': PARAM_NAMES_CN.get(param, param),
                'original_value': original_value,
                'qc_level': mark_level if mark_level > 0 else None,
                'qc_level_name': QC_LEVEL_NAMES.get(mark_level, '未知') if mark_level > 0 else '未知'
            })
    
    events_df = pd.DataFrame(events)
    if len(events_df) > 0:
        events_df = events_df.sort_values('time', ascending=False).reset_index(drop=True)
    return events_df
