import numpy as np
import pandas as pd
from scipy import signal
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go

from .plot_utils import setup_chinese_font
setup_chinese_font()


def welch_spectrum(hs_series: pd.Series, nperseg: int = 256, noverlap: int = None) -> Tuple[np.ndarray, np.ndarray]:
    if noverlap is None:
        noverlap = nperseg // 2
    valid_data = hs_series.dropna()
    if len(valid_data) < 512:
        raise ValueError("连续数据长度不足512点，无法计算频谱")
    values = valid_data.values
    times = valid_data.index
    if isinstance(times, pd.DatetimeIndex):
        dt = (times[1] - times[0]).total_seconds()
    else:
        dt = 60.0
    fs = 1.0 / dt
    freqs, psd = signal.welch(values, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann', detrend='linear')
    return freqs, psd


def compute_spectral_moments(freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
    positive = freqs > 0
    f = freqs[positive]
    s = psd[positive]
    df = np.diff(f, prepend=f[0])
    trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    m0 = trapz(s, f)
    m1 = trapz(s * f, f)
    m2 = trapz(s * f**2, f)
    m4 = trapz(s * f**4, f)
    return {'m0': m0, 'm1': m1, 'm2': m2, 'm4': m4}


def extract_wave_params(freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
    moments = compute_spectral_moments(freqs, psd)
    positive = freqs > 0
    peak_idx = np.argmax(psd[positive])
    fp = freqs[positive][peak_idx]
    Tp = 1.0 / fp if fp > 0 else np.nan
    Hm0 = 4.0 * np.sqrt(moments['m0']) if moments['m0'] > 0 else np.nan
    Tz = np.sqrt(moments['m0'] / moments['m2']) if moments['m2'] > 0 else np.nan
    epsilon = np.sqrt(1.0 - moments['m2']**2 / (moments['m0'] * moments['m4'])) if moments['m0'] > 0 and moments['m4'] > 0 else np.nan
    return {
        'fp': fp, 'Tp': Tp, 'Hm0': Hm0, 'Tz': Tz,
        'epsilon': epsilon, **moments
    }


def plot_wave_spectrum(freqs: np.ndarray, psd: np.ndarray, params: Dict = None) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freqs, psd, 'b-', linewidth=1.5)
    if params and 'fp' in params:
        ax.axvline(params['fp'], color='r', linestyle='--', label=f'fp={params["fp"]:.4f} Hz\nTp={params["Tp"]:.2f} s')
    ax.set_xlabel('频率 (Hz)')
    ax.set_ylabel('谱密度 S(f) (m²/Hz)')
    ax.set_title('波浪谱密度')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def compute_directional_spectrum(hs_series: pd.Series, wave_dir_series: pd.Series,
                                  nperseg: int = 256, noverlap: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if noverlap is None:
        noverlap = nperseg // 2
    combined = pd.concat([hs_series, wave_dir_series], axis=1).dropna()
    if len(combined) < 512:
        raise ValueError("连续数据长度不足512点，无法计算方向谱")
    hs = combined.iloc[:, 0].values
    wd = np.radians(combined.iloc[:, 1].values)
    times = combined.index
    if isinstance(times, pd.DatetimeIndex):
        dt = (times[1] - times[0]).total_seconds()
    else:
        dt = 60.0
    fs = 1.0 / dt
    freqs, psd = signal.welch(hs, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    directions = np.linspace(0, 2 * np.pi, 36)
    dir_spectrum = np.zeros((len(freqs), len(directions)))
    for i, f in enumerate(freqs):
        if f <= 0:
            continue
        cos_comp = np.cos(2 * wd)
        sin_comp = np.sin(2 * wd)
        _, cos_psd = signal.welch(hs * cos_comp, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
        _, sin_psd = signal.welch(hs * sin_comp, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
        a1 = cos_psd[i] / (psd[i] + 1e-10)
        b1 = sin_psd[i] / (psd[i] + 1e-10)
        mean_dir = np.arctan2(b1, a1)
        spread = np.sqrt(min(1.0, max(0, a1**2 + b1**2)))
        for j, theta in enumerate(directions):
            dir_spread = np.exp(-((theta - mean_dir)**2) / (2 * spread**2 + 1e-10))
            dir_spectrum[i, j] = psd[i] * dir_spread
    return freqs, directions, dir_spectrum


def plot_directional_spectrum(freqs: np.ndarray, directions: np.ndarray, dir_spec: np.ndarray) -> Figure:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    Theta, Freq = np.meshgrid(directions, freqs)
    mesh = ax.pcolormesh(Theta, Freq, dir_spec, cmap='jet', shading='auto')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(45)
    ax.set_ylabel('频率 (Hz)', labelpad=30)
    ax.set_title('波浪方向谱')
    plt.colorbar(mesh, ax=ax, label='谱密度 (m²/Hz/rad)', pad=0.1)
    plt.tight_layout()
    return fig


def analyze_wave_buoy(df: pd.DataFrame, buoy_id: str, qc_mask: Optional[pd.Series] = None) -> Dict:
    buoy_df = df[df['buoy_id'] == buoy_id].sort_values('time').set_index('time')
    result = {}
    hs = buoy_df['Hs'].copy()
    if qc_mask is not None:
        hs_qc = qc_mask[qc_mask['buoy_id'] == buoy_id].sort_values('time').set_index('time')
        hs[hs_qc['Hs'].isin([2, 3, 4])] = np.nan
    try:
        freqs, psd = welch_spectrum(hs)
        params = extract_wave_params(freqs, psd)
        result['freqs'] = freqs
        result['psd'] = psd
        result['params'] = params
        result['spectrum_fig'] = plot_wave_spectrum(freqs, psd, params)
        if 'wave_dir' in buoy_df.columns and buoy_df['wave_dir'].notna().sum() > 512:
            wd = buoy_df['wave_dir'].copy()
            if qc_mask is not None:
                wd[hs_qc['wave_dir'].isin([2, 3, 4])] = np.nan
            try:
                freqs_d, dirs_d, dir_spec = compute_directional_spectrum(hs, wd)
                result['dir_freqs'] = freqs_d
                result['dir_directions'] = dirs_d
                result['dir_spectrum'] = dir_spec
                result['dir_spectrum_fig'] = plot_directional_spectrum(freqs_d, dirs_d, dir_spec)
            except ValueError:
                pass
    except ValueError as e:
        result['error'] = str(e)
    return result


def compute_wave_monthly_stats(df: pd.DataFrame, buoy_id: str, qc_mask: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    buoy_df = df[df['buoy_id'] == buoy_id].sort_values('time').copy()
    if 'Hs' not in buoy_df.columns:
        return pd.DataFrame()
    hs = buoy_df['Hs'].copy()
    if qc_mask is not None:
        hs_qc = qc_mask[qc_mask['buoy_id'] == buoy_id].sort_values('time')
        hs = hs.reset_index(drop=True)
        hs_qc = hs_qc.reset_index(drop=True)
        if len(hs) == len(hs_qc):
            hs[hs_qc['Hs'].isin([2, 3, 4])] = np.nan
    buoy_df['Hs_valid'] = hs.values
    buoy_df['year_month'] = buoy_df['time'].dt.to_period('M')
    buoy_df['date'] = buoy_df['time'].dt.date
    monthly = buoy_df.groupby('year_month').agg(
        avg_hs=('Hs_valid', 'mean'),
        max_hs=('Hs_valid', 'max'),
        std_hs=('Hs_valid', 'std'),
        valid_days=('date', 'nunique')
    ).reset_index()
    over_2m_days = buoy_df[buoy_df['Hs_valid'] > 2].groupby('year_month')['date'].nunique()
    monthly['over_2m_days'] = monthly['year_month'].map(
        lambda ym: over_2m_days.get(ym, 0)
    )
    monthly['over_2m_ratio'] = monthly['over_2m_days'] / monthly['valid_days'].where(monthly['valid_days'] > 0, 1)
    monthly['year_month_str'] = monthly['year_month'].astype(str)
    monthly = monthly.rename(columns={
        'year_month_str': '月份',
        'avg_hs': '平均波高(m)',
        'max_hs': '最大波高(m)',
        'std_hs': '波高标准差(m)',
        'valid_days': '有效数据天数',
        'over_2m_ratio': '超过2m天数占比'
    })
    result = monthly[['月份', '平均波高(m)', '最大波高(m)', '波高标准差(m)', '有效数据天数', '超过2m天数占比']].copy()
    result = result.reset_index(drop=True)
    return result


def plot_wave_monthly_bar(monthly_stats: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    x_labels = monthly_stats['月份'].tolist()
    fig.add_trace(go.Bar(
        x=x_labels,
        y=monthly_stats['平均波高(m)'].tolist(),
        error_y=dict(
            type='data',
            array=monthly_stats['波高标准差(m)'].fillna(0).tolist(),
            visible=True,
            color='#888'
        ),
        marker_color='rgba(31, 119, 180, 0.8)',
        name='平均波高',
        hovertemplate='%{x}<br>平均波高: %{y:.3f} m<br>标准差: %{error_y.array:.3f} m<extra></extra>'
    ))
    fig.update_layout(
        title='月度平均波高统计',
        xaxis_title='月份',
        yaxis_title='平均波高 (m)',
        xaxis_tickangle=-45,
        template='plotly_white',
        height=500,
        showlegend=False
    )
    return fig
