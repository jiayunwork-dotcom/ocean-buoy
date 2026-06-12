import pandas as pd
import numpy as np
import xarray as xr
from io import BytesIO
from typing import Dict, List, Tuple, Optional

CSV_COLUMNS = [
    'time', 'buoy_id', 'wind_speed', 'wind_dir', 'air_temp',
    'pressure', 'Hs', 'Tz', 'wave_dir', 'SST', 'salinity',
    'current_speed', 'current_dir'
]

COLUMN_CF_MAPPING = {
    'wind_speed': ['wind_speed', 'ws', 'U10', 'wind_speed_10m'],
    'wind_dir': ['wind_dir', 'wd', 'wind_direction', 'wind_from_direction'],
    'air_temp': ['air_temp', 'at', 'T2', 'air_temperature', 'temperature_2m'],
    'pressure': ['pressure', 'pres', 'SLP', 'sea_level_pressure', 'psl'],
    'Hs': ['Hs', 'swh', 'significant_wave_height', 'wave_height'],
    'Tz': ['Tz', 't02', 'mean_wave_period', 'wave_period'],
    'wave_dir': ['wave_dir', 'mwd', 'mean_wave_direction', 'wave_direction'],
    'SST': ['SST', 'sst', 'sea_surface_temperature'],
    'salinity': ['salinity', 'sal', 'sss', 'sea_surface_salinity'],
    'current_speed': ['current_speed', 'cs', 'sea_water_speed'],
    'current_dir': ['current_dir', 'cd', 'sea_water_direction']
}


def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    for col in CSV_COLUMNS:
        if col not in df.columns and col != 'time' and col != 'buoy_id':
            df[col] = np.nan
    return df[CSV_COLUMNS]


def load_netcdf(file) -> pd.DataFrame:
    ds = xr.open_dataset(BytesIO(file.read()) if hasattr(file, 'read') else file)
    df = pd.DataFrame()
    
    if 'time' in ds.dims or 'time' in ds.coords:
        df['time'] = pd.to_datetime(ds['time'].values)
    elif 'TIME' in ds.dims or 'TIME' in ds.coords:
        df['time'] = pd.to_datetime(ds['TIME'].values)
    else:
        raise ValueError("NetCDF文件中未找到time坐标")
    
    if 'buoy_id' in ds.variables:
        df['buoy_id'] = ds['buoy_id'].values.astype(str)
    elif 'station' in ds.variables:
        df['buoy_id'] = ds['station'].values.astype(str)
    elif 'buoy' in ds.variables:
        df['buoy_id'] = ds['buoy'].values.astype(str)
    else:
        df['buoy_id'] = 'BUOY001'
    
    for std_name, possible_names in COLUMN_CF_MAPPING.items():
        found = False
        for name in possible_names:
            if name in ds.variables:
                df[std_name] = ds[name].values
                found = True
                break
        if not found:
            df[std_name] = np.nan
    
    return df[CSV_COLUMNS]


def load_file(file) -> pd.DataFrame:
    filename = file.name if hasattr(file, 'name') else str(file)
    if filename.endswith('.csv'):
        return load_csv(file)
    elif filename.endswith('.nc') or filename.endswith('.netcdf'):
        return load_netcdf(file)
    else:
        raise ValueError(f"不支持的文件格式: {filename}")


def merge_buoy_data(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(dataframes, ignore_index=True)
    merged = merged.sort_values(['buoy_id', 'time']).reset_index(drop=True)
    return merged


def get_data_overview(df: pd.DataFrame) -> Dict:
    overview = {}
    for buoy_id in df['buoy_id'].unique():
        buoy_df = df[df['buoy_id'] == buoy_id]
        param_stats = {}
        for col in CSV_COLUMNS[2:]:
            total = len(buoy_df)
            missing = buoy_df[col].isna().sum()
            param_stats[col] = {
                'total': total,
                'missing': missing,
                'missing_rate': missing / total if total > 0 else 0
            }
        overview[buoy_id] = {
            'time_start': buoy_df['time'].min(),
            'time_end': buoy_df['time'].max(),
            'record_count': len(buoy_df),
            'parameters': param_stats
        }
    return overview


def get_buoy_locations() -> Optional[Dict[str, Tuple[float, float]]]:
    return None
