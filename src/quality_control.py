import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats

QC_UNCHECKED = 0
QC_GOOD = 1
QC_SUSPECT = 2
QC_ERROR = 3
QC_MISSING = 4
QC_INTERPOLATED = 5

PARAMETERS = [
    'wind_speed', 'wind_dir', 'air_temp', 'pressure',
    'Hs', 'Tz', 'wave_dir', 'SST', 'salinity',
    'current_speed', 'current_dir'
]

DEFAULT_RANGE_THRESHOLDS = {
    'wind_speed': {'min': 0, 'max': 60},
    'wind_dir': {'min': 0, 'max': 360},
    'air_temp': {'min': -40, 'max': 50},
    'pressure': {'min': 870, 'max': 1080},
    'Hs': {'min': 0, 'max': 20},
    'Tz': {'min': 0, 'max': 30},
    'wave_dir': {'min': 0, 'max': 360},
    'SST': {'min': -5, 'max': 40},
    'salinity': {'min': 0, 'max': 42},
    'current_speed': {'min': 0, 'max': 10},
    'current_dir': {'min': 0, 'max': 360}
}

DEFAULT_GRADIENT_THRESHOLDS = {
    'wind_speed': 15.0,
    'air_temp': 5.0,
    'pressure': 10.0,
    'Hs': 3.0,
    'SST': 3.0,
    'salinity': 2.0,
    'current_speed': 2.0
}


@dataclass
class QCResult:
    qc_codes: pd.DataFrame = None
    manual_overrides: Dict[Tuple[str, pd.Timestamp, str], Tuple[int, str]] = field(default_factory=dict)
    level_stats: Dict = field(default_factory=dict)
    
    def get_final_code(self, buoy_id: str, time: pd.Timestamp, param: str) -> int:
        key = (buoy_id, time, param)
        if key in self.manual_overrides:
            return self.manual_overrides[key][0]
        if self.qc_codes is not None:
            mask = (self.qc_codes['buoy_id'] == buoy_id) & (self.qc_codes['time'] == time)
            if mask.any():
                return int(self.qc_codes.loc[mask, param].values[0])
        return QC_UNCHECKED


def init_qc_codes(df: pd.DataFrame) -> pd.DataFrame:
    qc_df = df[['time', 'buoy_id']].copy()
    for param in PARAMETERS:
        qc_df[param] = QC_UNCHECKED
    return qc_df


def qc_level1_range(df: pd.DataFrame, qc_df: pd.DataFrame, thresholds: Dict = None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = DEFAULT_RANGE_THRESHOLDS
    qc_df = qc_df.copy()
    for param, thresh in thresholds.items():
        if param not in df.columns:
            continue
        mask_missing = df[param].isna()
        mask_error = (~mask_missing) & ((df[param] < thresh['min']) | (df[param] > thresh['max']))
        qc_df[param] = np.where(mask_missing, QC_MISSING,
                               np.where(mask_error, QC_ERROR, qc_df[param]))
    return qc_df


def qc_level2_temporal(df: pd.DataFrame, qc_df: pd.DataFrame, thresholds: Dict = None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = DEFAULT_GRADIENT_THRESHOLDS
    qc_df = qc_df.copy()
    for buoy_id in df['buoy_id'].unique():
        buoy_mask = df['buoy_id'] == buoy_id
        buoy_df = df.loc[buoy_mask].sort_values('time')
        buoy_qc = qc_df.loc[buoy_mask].sort_values('time')
        if len(buoy_df) < 2:
            continue
        for param, grad_thresh in thresholds.items():
            if param not in df.columns:
                continue
            values = buoy_df[param].values
            times = buoy_df['time'].values
            qc_values = buoy_qc[param].values.copy()
            for i in range(1, len(values)):
                if np.isnan(values[i]) or np.isnan(values[i-1]):
                    continue
                if qc_values[i] == QC_MISSING or qc_values[i] == QC_ERROR:
                    continue
                dt_hours = (times[i] - times[i-1]).astype('timedelta64[s]').astype(float) / 3600.0
                if dt_hours <= 0:
                    continue
                change_rate = abs(values[i] - values[i-1]) / dt_hours
                if change_rate > grad_thresh:
                    qc_values[i] = QC_SUSPECT
            qc_df.loc[buoy_mask, param] = qc_values
    return qc_df


def qc_level3_internal(df: pd.DataFrame, qc_df: pd.DataFrame) -> pd.DataFrame:
    qc_df = qc_df.copy()
    for buoy_id in df['buoy_id'].unique():
        buoy_mask = df['buoy_id'] == buoy_id
        buoy_df = df.loc[buoy_mask]
        buoy_qc = qc_df.loc[buoy_mask].copy()
        if 'wind_speed' in buoy_df.columns and 'Hs' in buoy_df.columns:
            wind_zero = (buoy_df['wind_speed'] == 0) | (buoy_df['wind_speed'] < 0.5)
            wave_high = buoy_df['Hs'] > 2.0
            suspect_mask = wind_zero & wave_high & buoy_df['Hs'].notna() & buoy_df['wind_speed'].notna()
            for idx in buoy_qc.index[suspect_mask]:
                if buoy_qc.loc[idx, 'Hs'] not in [QC_ERROR, QC_MISSING]:
                    buoy_qc.loc[idx, 'Hs'] = QC_SUSPECT
                if buoy_qc.loc[idx, 'wind_speed'] not in [QC_ERROR, QC_MISSING]:
                    buoy_qc.loc[idx, 'wind_speed'] = QC_SUSPECT
        qc_df.loc[buoy_mask] = buoy_qc
    return qc_df


def qc_level4_climatology(df: pd.DataFrame, qc_df: pd.DataFrame,
                          climatology: Optional[Dict] = None) -> pd.DataFrame:
    qc_df = qc_df.copy()
    if climatology is None:
        climatology = {}
        for buoy_id in df['buoy_id'].unique():
            buoy_df = df[df['buoy_id'] == buoy_id].copy()
            buoy_df['month'] = buoy_df['time'].dt.month
            climatology[buoy_id] = {}
            for param in PARAMETERS:
                if param not in buoy_df.columns:
                    continue
                clim = buoy_df.groupby('month')[param].agg(['mean', 'std'])
                climatology[buoy_id][param] = clim
    for buoy_id in df['buoy_id'].unique():
        if buoy_id not in climatology:
            continue
        buoy_mask = df['buoy_id'] == buoy_id
        buoy_df = df.loc[buoy_mask].copy()
        buoy_df['month'] = buoy_df['time'].dt.month
        buoy_qc = qc_df.loc[buoy_mask].copy()
        for param in PARAMETERS:
            if param not in climatology.get(buoy_id, {}):
                continue
            clim = climatology[buoy_id][param]
            months = buoy_df['month'].values
            values = buoy_df[param].values
            qc_values = buoy_qc[param].values.copy()
            for i in range(len(values)):
                if np.isnan(values[i]) or qc_values[i] in [QC_ERROR, QC_MISSING]:
                    continue
                month = months[i]
                if month in clim.index:
                    mean = clim.loc[month, 'mean']
                    std = clim.loc[month, 'std']
                    if not np.isnan(mean) and not np.isnan(std) and std > 0:
                        if abs(values[i] - mean) > 4 * std:
                            qc_values[i] = QC_SUSPECT
            qc_df.loc[buoy_mask, param] = qc_values
    return qc_df


def qc_level5_spike(df: pd.DataFrame, qc_df: pd.DataFrame, sigma: float = 3.0) -> pd.DataFrame:
    qc_df = qc_df.copy()
    for buoy_id in df['buoy_id'].unique():
        buoy_mask = df['buoy_id'] == buoy_id
        buoy_df = df.loc[buoy_mask].sort_values('time')
        buoy_qc = qc_df.loc[buoy_mask].sort_values('time').copy()
        for param in PARAMETERS:
            if param not in buoy_df.columns:
                continue
            values = buoy_df[param].values
            qc_values = buoy_qc[param].values.copy()
            n = len(values)
            if n < 3:
                continue
            for i in range(1, n - 1):
                if np.isnan(values[i]) or np.isnan(values[i-1]) or np.isnan(values[i+1]):
                    continue
                if qc_values[i] in [QC_ERROR, QC_MISSING]:
                    continue
                diff_prev = abs(values[i] - values[i-1])
                diff_next = abs(values[i] - values[i+1])
                diff_neighbors = abs(values[i-1] - values[i+1])
                local_std = np.nanstd(values[max(0,i-10):min(n,i+11)])
                if local_std <= 0:
                    continue
                if diff_prev > sigma * local_std and diff_next > sigma * local_std and diff_neighbors < sigma * local_std:
                    qc_values[i] = QC_ERROR
            qc_df.loc[buoy_qc.index, param] = qc_values
    return qc_df


def qc_level6_stuck(df: pd.DataFrame, qc_df: pd.DataFrame, n_consecutive: int = 6) -> pd.DataFrame:
    qc_df = qc_df.copy()
    for buoy_id in df['buoy_id'].unique():
        buoy_mask = df['buoy_id'] == buoy_id
        buoy_df = df.loc[buoy_mask].sort_values('time')
        buoy_qc = qc_df.loc[buoy_mask].sort_values('time').copy()
        for param in PARAMETERS:
            if param not in buoy_df.columns:
                continue
            values = buoy_df[param].values
            qc_values = buoy_qc[param].values.copy()
            n = len(values)
            if n < n_consecutive:
                continue
            i = 0
            while i < n:
                if np.isnan(values[i]):
                    i += 1
                    continue
                j = i
                while j < n and not np.isnan(values[j]) and values[j] == values[i]:
                    j += 1
                if j - i >= n_consecutive:
                    for k in range(i, j):
                        if qc_values[k] not in [QC_ERROR, QC_MISSING]:
                            qc_values[k] = QC_SUSPECT
                i = j
            qc_df.loc[buoy_qc.index, param] = qc_values
    return qc_df


def qc_level7_spatial(df: pd.DataFrame, qc_df: pd.DataFrame,
                      buoy_locations: Optional[Dict[str, Tuple[float, float]]] = None,
                      max_distance_km: float = 100.0) -> pd.DataFrame:
    if buoy_locations is None:
        return qc_df
    buoy_ids = df['buoy_id'].unique()
    if len(buoy_ids) < 2:
        return qc_df
    qc_df = qc_df.copy()
    for param in PARAMETERS:
        if param not in df.columns:
            continue
        pivot = df.pivot_table(index='time', columns='buoy_id', values=param)
        qc_pivot = qc_df.pivot_table(index='time', columns='buoy_id', values=param)
        for time in pivot.index:
            row = pivot.loc[time]
            qc_row = qc_pivot.loc[time].copy()
            valid_buoys = row.dropna().index.tolist()
            if len(valid_buoys) < 2:
                continue
            for i, b1 in enumerate(valid_buoys):
                if qc_row[b1] in [QC_ERROR, QC_MISSING]:
                    continue
                neighbors = []
                for b2 in valid_buoys:
                    if b1 == b2:
                        continue
                    if b1 in buoy_locations and b2 in buoy_locations:
                        dist = haversine_distance(buoy_locations[b1], buoy_locations[b2])
                        if dist <= max_distance_km:
                            neighbors.append(b2)
                if not neighbors:
                    continue
                neighbor_values = [row[b] for b in neighbors]
                local_std = np.std(neighbor_values) if len(neighbor_values) >= 2 else np.std(pivot[neighbors].dropna().values)
                if local_std <= 0:
                    continue
                neighbor_mean = np.mean(neighbor_values)
                if abs(row[b1] - neighbor_mean) > 3 * local_std:
                    qc_row[b1] = QC_SUSPECT
            qc_pivot.loc[time] = qc_row
        for buoy_id in buoy_ids:
            if buoy_id in qc_pivot.columns:
                mask = qc_df['buoy_id'] == buoy_id
                times = qc_df.loc[mask, 'time'].values
                codes = []
                for t in times:
                    if t in qc_pivot.index:
                        codes.append(int(qc_pivot.loc[t, buoy_id]))
                    else:
                        codes.append(int(qc_df.loc[mask & (qc_df['time'] == t), param].values[0]))
                qc_df.loc[mask, param] = codes
    return qc_df


def haversine_distance(loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
    lat1, lon1 = np.radians(loc1)
    lat2, lon2 = np.radians(loc2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def run_full_qc(df: pd.DataFrame,
                range_thresholds: Dict = None,
                gradient_thresholds: Dict = None,
                climatology: Optional[Dict] = None,
                buoy_locations: Optional[Dict[str, Tuple[float, float]]] = None,
                spike_sigma: float = 3.0,
                stuck_n: int = 6) -> QCResult:
    qc_df = init_qc_codes(df)
    level_stats = {}
    
    qc_df = qc_level1_range(df, qc_df, range_thresholds)
    level_stats[1] = count_qc_codes(qc_df)
    
    qc_df = qc_level2_temporal(df, qc_df, gradient_thresholds)
    level_stats[2] = count_qc_codes(qc_df)
    
    qc_df = qc_level3_internal(df, qc_df)
    level_stats[3] = count_qc_codes(qc_df)
    
    qc_df = qc_level4_climatology(df, qc_df, climatology)
    level_stats[4] = count_qc_codes(qc_df)
    
    qc_df = qc_level5_spike(df, qc_df, spike_sigma)
    level_stats[5] = count_qc_codes(qc_df)
    
    qc_df = qc_level6_stuck(df, qc_df, stuck_n)
    level_stats[6] = count_qc_codes(qc_df)
    
    qc_df = qc_level7_spatial(df, qc_df, buoy_locations)
    level_stats[7] = count_qc_codes(qc_df)
    
    for param in PARAMETERS:
        if param in qc_df.columns:
            unmarked = qc_df[param] == QC_UNCHECKED
            qc_df.loc[unmarked & df[param].notna(), param] = QC_GOOD
    
    return QCResult(qc_codes=qc_df, manual_overrides={}, level_stats=level_stats)


def count_qc_codes(qc_df: pd.DataFrame) -> Dict[str, Dict[int, int]]:
    counts = {}
    for param in PARAMETERS:
        if param in qc_df.columns:
            counts[param] = qc_df[param].value_counts().to_dict()
    return counts
