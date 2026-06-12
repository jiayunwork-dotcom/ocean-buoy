import numpy as np
import pandas as pd
from scipy import signal, optimize
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .plot_utils import setup_chinese_font
setup_chinese_font()

TIDAL_CONSTITUENTS = {
    'M2': 12.4206, 'S2': 12.0, 'N2': 12.6583, 'K2': 11.9672,
    'K1': 23.9345, 'O1': 25.8193, 'P1': 24.0659, 'Q1': 26.8683
}


def polar_to_cartesian(speed: np.ndarray, direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dir_rad = np.radians(direction)
    u = speed * np.sin(dir_rad)
    v = speed * np.cos(dir_rad)
    return u, v


def cartesian_to_polar(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(u**2 + v**2)
    direction = np.degrees(np.arctan2(u, v)) % 360
    return speed, direction


def tidal_analysis(times: pd.DatetimeIndex, u: np.ndarray, v: np.ndarray,
                   constituents: List[str] = None) -> Dict:
    if constituents is None:
        constituents = list(TIDAL_CONSTITUENTS.keys())
    duration_days = (times[-1] - times[0]).total_seconds() / 86400.0
    if duration_days < 30:
        raise ValueError(f"数据时长仅{duration_days:.1f}天，至少需要30天才能进行潮流调和分析")
    t_seconds = (times - times[0]).total_seconds().values
    n = len(times)
    n_const = len(constituents)
    A = np.zeros((n, 2 + 2 * n_const))
    A[:, 0] = 1.0
    A[:, 1] = t_seconds / 86400.0
    for i, const_name in enumerate(constituents):
        period_hours = TIDAL_CONSTITUENTS[const_name]
        omega = 2 * np.pi / (period_hours * 3600.0)
        A[:, 2 + 2 * i] = np.cos(omega * t_seconds)
        A[:, 3 + 2 * i] = np.sin(omega * t_seconds)
    valid_u = ~np.isnan(u)
    valid_v = ~np.isnan(v)
    result = {'u': {}, 'v': {}}
    if valid_u.sum() > 2 * (n_const + 1):
        coeffs_u, _, _, _ = np.linalg.lstsq(A[valid_u], u[valid_u], rcond=None)
        result['u']['residual'] = u - A @ coeffs_u
        result['u']['mean'] = coeffs_u[0]
        result['u']['trend'] = coeffs_u[1]
        for i, const_name in enumerate(constituents):
            a = coeffs_u[2 + 2 * i]
            b = coeffs_u[3 + 2 * i]
            amplitude = np.sqrt(a**2 + b**2)
            phase = np.degrees(np.arctan2(-b, a)) % 360
            result['u'][const_name] = {'amplitude': amplitude, 'phase': phase, 'a': a, 'b': b}
    if valid_v.sum() > 2 * (n_const + 1):
        coeffs_v, _, _, _ = np.linalg.lstsq(A[valid_v], v[valid_v], rcond=None)
        result['v']['residual'] = v - A @ coeffs_v
        result['v']['mean'] = coeffs_v[0]
        result['v']['trend'] = coeffs_v[1]
        for i, const_name in enumerate(constituents):
            a = coeffs_v[2 + 2 * i]
            b = coeffs_v[3 + 2 * i]
            amplitude = np.sqrt(a**2 + b**2)
            phase = np.degrees(np.arctan2(-b, a)) % 360
            result['v'][const_name] = {'amplitude': amplitude, 'phase': phase, 'a': a, 'b': b}
    ellipses = {}
    for const_name in constituents:
        if const_name in result['u'] and const_name in result['v']:
            u_amp = result['u'][const_name]['amplitude']
            v_amp = result['v'][const_name]['amplitude']
            u_phase = np.radians(result['u'][const_name]['phase'])
            v_phase = np.radians(result['v'][const_name]['phase'])
            u_a = result['u'][const_name]['a']
            u_b = result['u'][const_name]['b']
            v_a = result['v'][const_name]['a']
            v_b = result['v'][const_name]['b']
            M = np.array([[u_a, u_b], [v_a, v_b]])
            eigvals, _ = np.linalg.eig(M @ M.T)
            eigvals = np.sort(eigvals)[::-1]
            semi_major = np.sqrt(max(0, eigvals[0]))
            semi_minor = np.sqrt(max(0, eigvals[1]))
            inclination = 0.5 * np.degrees(np.arctan2(2 * (u_a * v_a + u_b * v_b),
                                                      u_a**2 + u_b**2 - v_a**2 - v_b**2))
            eccentricity = semi_minor / semi_major if semi_major > 0 else 0
            ellipses[const_name] = {
                'semi_major': semi_major, 'semi_minor': semi_minor,
                'inclination': inclination, 'eccentricity': eccentricity
            }
    result['ellipses'] = ellipses
    return result


def plot_tidal_ellipses(ellipses: Dict) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 8))
    theta = np.linspace(0, 2 * np.pi, 100)
    colors = plt.cm.tab10(np.linspace(0, 1, len(ellipses)))
    for i, (name, params) in enumerate(ellipses.items()):
        a = params['semi_major']
        b = params['semi_minor']
        inc = np.radians(params['inclination'])
        x = a * np.cos(theta) * np.cos(inc) - b * np.sin(theta) * np.sin(inc)
        y = a * np.cos(theta) * np.sin(inc) + b * np.sin(theta) * np.cos(inc)
        ax.plot(x, y, color=colors[i], linewidth=2, label=f'{name}')
        ax.plot([0, params['semi_major'] * np.cos(inc)],
                [0, params['semi_major'] * np.sin(inc)],
                '--', color=colors[i], alpha=0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('u (东向流速, m/s)')
    ax.set_ylabel('v (北向流速, m/s)')
    ax.set_title('潮流椭圆')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def current_rose(speed: np.ndarray, direction: np.ndarray, n_sectors: int = 16,
                 speed_bins: List[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if speed_bins is None:
        speed_bins = [0, 0.2, 0.5, 1.0, 1.5, np.inf]
    valid = ~(np.isnan(speed) | np.isnan(direction))
    speed = speed[valid]
    direction = direction[valid]
    sector_edges = np.linspace(-11.25, 360 + 11.25, n_sectors + 1)
    sector_centers = np.linspace(0, 360, n_sectors, endpoint=False)
    counts = np.zeros((n_sectors, len(speed_bins) - 1))
    for i in range(n_sectors):
        mask = ((direction >= sector_edges[i]) & (direction < sector_edges[i + 1])) | \
               ((direction + 360 >= sector_edges[i]) & (direction + 360 < sector_edges[i + 1]))
        for j in range(len(speed_bins) - 1):
            counts[i, j] = np.sum(mask & (speed >= speed_bins[j]) & (speed < speed_bins[j + 1]))
    total = np.sum(counts)
    if total > 0:
        percentages = counts / total * 100
    else:
        percentages = counts
    return sector_centers, percentages, speed_bins


def plot_current_rose(sector_centers: np.ndarray, percentages: np.ndarray, speed_bins: List[float]) -> Figure:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    theta = np.radians(sector_centers)
    width = 2 * np.pi / len(sector_centers)
    colors = plt.cm.YlOrRd(np.linspace(0.2, 1, percentages.shape[1]))
    bottom = np.zeros(len(theta))
    for j in range(percentages.shape[1]):
        ax.bar(theta, percentages[:, j], width=width, bottom=bottom,
               color=colors[j], label=f'{speed_bins[j]:.1f}-{speed_bins[j+1]:.1f} m/s')
        bottom += percentages[:, j]
    ax.set_title('流速流向玫瑰图')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    return fig


def lowpass_filter(data: np.ndarray, cutoff_hours: float = 25, dt_minutes: float = 60) -> np.ndarray:
    nyquist = 0.5 / (dt_minutes / 60.0)
    cutoff = 1.0 / cutoff_hours
    normalized_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype='low')
    valid = ~np.isnan(data)
    filtered = np.full_like(data, np.nan)
    if valid.sum() > 10:
        interpolated = np.interp(np.arange(len(data)), np.where(valid)[0], data[valid])
        filtered_valid = signal.filtfilt(b, a, interpolated)
        filtered[valid] = filtered_valid[valid]
    return filtered


def plot_residual_current(times: pd.DatetimeIndex, u_res: np.ndarray, v_res: np.ndarray) -> Figure:
    u_filt = lowpass_filter(u_res)
    v_filt = lowpass_filter(v_res)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes[0].plot(times, u_filt, 'b-', linewidth=1)
    axes[0].set_ylabel('u (东向余流, m/s)')
    axes[0].set_title('余流时序 (低通滤波，截止25小时)')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(times, v_filt, 'r-', linewidth=1)
    axes[1].set_ylabel('v (北向余流, m/s)')
    axes[1].grid(True, alpha=0.3)
    step = max(1, len(times) // 50)
    idx = np.arange(0, len(times), step)
    valid = ~(np.isnan(u_filt[idx]) | np.isnan(v_filt[idx]))
    axes[2].quiver(times[idx][valid], np.zeros_like(times[idx][valid]),
                   u_filt[idx][valid], v_filt[idx][valid], scale=5)
    axes[2].set_ylabel('余流矢量')
    axes[2].set_yticks([])
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def analyze_current_buoy(df: pd.DataFrame, buoy_id: str, qc_mask: Optional[pd.DataFrame] = None) -> Dict:
    buoy_df = df[df['buoy_id'] == buoy_id].sort_values('time')
    times = buoy_df['time'].values
    result = {}
    cs = buoy_df['current_speed'].values.copy()
    cd = buoy_df['current_dir'].values.copy()
    if qc_mask is not None:
        qc_buoy = qc_mask[qc_mask['buoy_id'] == buoy_id].sort_values('time')
        bad = qc_buoy['current_speed'].isin([2, 3, 4]).values
        cs[bad] = np.nan
        cd[bad] = np.nan
    u, v = polar_to_cartesian(cs, cd)
    duration_days = (times[-1] - times[0]).astype('timedelta64[s]').astype(float) / 86400.0
    if duration_days >= 30:
        try:
            tidal_result = tidal_analysis(pd.DatetimeIndex(times), u, v)
            result['tidal'] = tidal_result
            result['tidal_ellipses_fig'] = plot_tidal_ellipses(tidal_result['ellipses'])
            if 'u' in tidal_result and 'v' in tidal_result and 'residual' in tidal_result['u']:
                result['residual_fig'] = plot_residual_current(
                    pd.DatetimeIndex(times), tidal_result['u']['residual'], tidal_result['v']['residual'])
        except (ValueError, np.linalg.LinAlgError) as e:
            result['tidal_error'] = str(e)
    else:
        result['tidal_warning'] = f"数据时长仅{duration_days:.1f}天，潮流调和分析需要至少30天数据"
    valid_count = np.sum(~(np.isnan(cs) | np.isnan(cd)))
    if valid_count > 100:
        sectors, percentages, bins = current_rose(cs, cd)
        result['rose_sectors'] = sectors
        result['rose_percentages'] = percentages
        result['rose_bins'] = bins
        result['current_rose_fig'] = plot_current_rose(sectors, percentages, bins)
    return result
