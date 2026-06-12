import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.cm as cm

from .plot_utils import setup_chinese_font
setup_chinese_font()


def wind_rose(wind_speed: np.ndarray, wind_dir: np.ndarray, n_sectors: int = 16,
              speed_bins: List[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if speed_bins is None:
        speed_bins = [0, 2, 5, 10, 15, np.inf]
    valid = ~(np.isnan(wind_speed) | np.isnan(wind_dir))
    wind_speed = wind_speed[valid]
    wind_dir = wind_dir[valid]
    sector_edges = np.linspace(-11.25, 360 + 11.25, n_sectors + 1)
    sector_centers = np.linspace(0, 360, n_sectors, endpoint=False)
    counts = np.zeros((n_sectors, len(speed_bins) - 1))
    for i in range(n_sectors):
        mask = ((wind_dir >= sector_edges[i]) & (wind_dir < sector_edges[i + 1])) | \
               ((wind_dir + 360 >= sector_edges[i]) & (wind_dir + 360 < sector_edges[i + 1]))
        for j in range(len(speed_bins) - 1):
            counts[i, j] = np.sum(mask & (wind_speed >= speed_bins[j]) & (wind_speed < speed_bins[j + 1]))
    total = np.sum(counts)
    if total > 0:
        percentages = counts / total * 100
    else:
        percentages = counts
    return sector_centers, percentages, np.array(speed_bins)


def plot_wind_rose(sector_centers: np.ndarray, percentages: np.ndarray, speed_bins: np.ndarray) -> Figure:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    theta = np.radians(sector_centers)
    width = 2 * np.pi / len(sector_centers)
    colors = plt.cm.Blues(np.linspace(0.3, 1, percentages.shape[1]))
    bottom = np.zeros(len(theta))
    for j in range(percentages.shape[1]):
        ax.bar(theta, percentages[:, j], width=width, bottom=bottom,
               color=colors[j], label=f'{speed_bins[j]:.0f}-{speed_bins[j+1]:.0f} m/s')
        bottom += percentages[:, j]
    ax.set_title('风速风向玫瑰图')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    return fig


def detect_low_pressure(times: pd.DatetimeIndex, pressure: np.ndarray,
                        drop_threshold: float = 10.0, hours_window: float = 6.0) -> List[Dict]:
    events = []
    dt_hours = np.diff(times).astype('timedelta64[s]').astype(float) / 3600.0
    for i in range(len(times)):
        window_start = times[i] - pd.Timedelta(hours=hours_window)
        window_mask = (times >= window_start) & (times <= times[i])
        if window_mask.sum() < 2:
            continue
        window_p = pressure[window_mask]
        if np.any(np.isnan(window_p)):
            continue
        drop = window_p[0] - window_p[-1]
        if drop >= drop_threshold:
            events.append({
                'time': times[i],
                'pressure_drop': drop,
                'pressure_min': np.min(window_p),
                'pressure_start': window_p[0]
            })
    return events


def plot_pressure_trend(times: pd.DatetimeIndex, pressure: np.ndarray,
                        low_pressure_events: List[Dict] = None) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, pressure, 'b-', linewidth=1, label='气压')
    if low_pressure_events:
        for event in low_pressure_events:
            ax.axvline(event['time'], color='r', linestyle='--', alpha=0.5)
            ax.annotate(f"低压过境\n{event['pressure_drop']:.1f}hPa",
                        (event['time'], event['pressure_min']),
                        textcoords="offset points", xytext=(0, -30), ha='center',
                        fontsize=8, color='red')
    ax.set_xlabel('时间')
    ax.set_ylabel('气压 (hPa)')
    ax.set_title('气压时序趋势')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_diurnal_variation(df: pd.DataFrame, param: str) -> Figure:
    df = df.copy()
    df['hour'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    monthly_diurnal = df.groupby(['month', 'hour'])[param].mean().unstack()
    fig, ax = plt.subplots(figsize=(12, 6))
    months = sorted(df['month'].unique())
    colors = cm.tab20(np.linspace(0, 1, len(months)))
    for i, month in enumerate(months):
        if month in monthly_diurnal.index:
            ax.plot(monthly_diurnal.columns, monthly_diurnal.loc[month],
                    color=colors[i], marker='o', label=f'{month}月', markersize=4)
    overall = df.groupby('hour')[param].mean()
    ax.plot(overall.index, overall.values, 'k-', linewidth=3, label='全年平均')
    ax.set_xlabel('小时')
    ax.set_xticks(range(0, 24, 2))
    param_names = {'air_temp': '气温 (℃)', 'SST': '海表温度 (℃)', 'pressure': '气压 (hPa)'}
    ax.set_ylabel(param_names.get(param, param))
    ax.set_title(f'{param_names.get(param, param)}日变化')
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_multi_buoy_comparison(df: pd.DataFrame, buoy_ids: List[str], param: str) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(buoy_ids)))
    for i, buoy_id in enumerate(buoy_ids):
        buoy_df = df[df['buoy_id'] == buoy_id].sort_values('time')
        ax.plot(buoy_df['time'], buoy_df[param], color=colors[i], label=buoy_id, linewidth=1, alpha=0.8)
    param_names = {'wind_speed': '风速 (m/s)', 'air_temp': '气温 (℃)', 'pressure': '气压 (hPa)',
                   'Hs': '有效波高 (m)', 'SST': '海表温度 (℃)', 'Tz': '平均波周期 (s)'}
    ax.set_xlabel('时间')
    ax.set_ylabel(param_names.get(param, param))
    ax.set_title(f'多浮标{param_names.get(param, param)}对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def compute_monthly_means_table(df: pd.DataFrame, buoy_ids: List[str], param: str) -> pd.DataFrame:
    df = df.copy()
    df['month'] = df['time'].dt.month
    df_filtered = df[df['buoy_id'].isin(buoy_ids)]
    table = df_filtered.pivot_table(index='month', columns='buoy_id', values=param, aggfunc='mean')
    return table


def compute_correlation_matrix(df: pd.DataFrame, buoy_ids: List[str], param: str) -> pd.DataFrame:
    pivot = df[df['buoy_id'].isin(buoy_ids)].pivot_table(index='time', columns='buoy_id', values=param)
    return pivot.corr()


def analyze_meteorology_buoy(df: pd.DataFrame, buoy_id: str, qc_mask: Optional[pd.DataFrame] = None) -> Dict:
    buoy_df = df[df['buoy_id'] == buoy_id].sort_values('time')
    times = buoy_df['time'].values
    result = {}
    ws = buoy_df['wind_speed'].values.copy()
    wd = buoy_df['wind_dir'].values.copy()
    if qc_mask is not None:
        qc_buoy = qc_mask[qc_mask['buoy_id'] == buoy_id].sort_values('time')
        ws[qc_buoy['wind_speed'].isin([3, 4]).values] = np.nan
        wd[qc_buoy['wind_dir'].isin([3, 4]).values] = np.nan
    valid_wind = np.sum(~(np.isnan(ws) | np.isnan(wd)))
    if valid_wind > 50:
        sectors, percentages, bins = wind_rose(ws, wd)
        result['wind_rose'] = plot_wind_rose(sectors, percentages, bins)
    pres = buoy_df['pressure'].values.copy()
    if qc_mask is not None:
        qc_buoy = qc_mask[qc_mask['buoy_id'] == buoy_id].sort_values('time')
        pres[qc_buoy['pressure'].isin([3, 4]).values] = np.nan
    if np.sum(~np.isnan(pres)) > 50:
        low_p_events = detect_low_pressure(pd.DatetimeIndex(times), pres)
        result['pressure_plot'] = plot_pressure_trend(pd.DatetimeIndex(times), pres, low_p_events)
        result['low_pressure_events'] = low_p_events
    if 'air_temp' in buoy_df.columns and buoy_df['air_temp'].notna().sum() > 50:
        at = buoy_df['air_temp'].values.copy()
        if qc_mask is not None:
            qc_buoy = qc_mask[qc_mask['buoy_id'] == buoy_id].sort_values('time')
            at[qc_buoy['air_temp'].isin([3, 4]).values] = np.nan
        temp_df = buoy_df.copy()
        temp_df['air_temp'] = at
        result['temp_diurnal'] = plot_diurnal_variation(temp_df, 'air_temp')
    return result
