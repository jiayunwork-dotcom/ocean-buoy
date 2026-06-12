import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.data_loader import load_file, get_data_overview
from src.quality_control import (
    run_full_qc, qc_level1_range, qc_level2_temporal, qc_level3_internal,
    qc_level4_climatology, qc_level5_spike, qc_level6_stuck,
    init_qc_codes, QC_GOOD, QC_SUSPECT, QC_ERROR, QC_MISSING, QC_UNCHECKED,
    PARAMETERS, DEFAULT_RANGE_THRESHOLDS, DEFAULT_GRADIENT_THRESHOLDS
)

df = pd.read_csv('sample_buoy_data.csv')
df['time'] = pd.to_datetime(df['time'])

print("=" * 60)
print("数据基本信息")
print("=" * 60)
print(f"记录数: {len(df)}")
print(f"时间范围: {df['time'].min()} ~ {df['time'].max()}")
print(f"浮标: {df['buoy_id'].unique()}")

print("\n" + "=" * 60)
print("各参数统计")
print("=" * 60)
for param in PARAMETERS:
    if param in df.columns:
        print(f"\n{param}:")
        print(f"  min={df[param].min():.4f}, max={df[param].max():.4f}")
        print(f"  mean={df[param].mean():.4f}, std={df[param].std():.4f}")
        print(f"  缺测数: {df[param].isna().sum()}")

print("\n" + "=" * 60)
print("逐分钟变化率统计 (1分钟变化)")
print("=" * 60)
for param in ['wind_speed', 'air_temp', 'pressure', 'Hs', 'SST', 'salinity']:
    if param in df.columns:
        values = df[param].values
        diffs = np.abs(np.diff(values))
        valid = ~np.isnan(diffs)
        if np.any(valid):
            print(f"\n{param}:")
            print(f"  1分钟变化 - min={np.nanmin(diffs):.4f}, max={np.nanmax(diffs):.4f}")
            print(f"  1分钟变化 - mean={np.nanmean(diffs):.4f}, std={np.nanstd(diffs):.4f}")
            grad_per_hour = diffs * 60  # 每小时变化率
            print(f"  换算每小时变化 - 95分位={np.nanpercentile(grad_per_hour, 95):.4f}")
            print(f"  换算每小时变化 - 99分位={np.nanpercentile(grad_per_hour, 99):.4f}")
            threshold = DEFAULT_GRADIENT_THRESHOLDS.get(param, None)
            if threshold:
                count = np.sum(grad_per_hour[valid] > threshold)
                print(f"  超过阈值({threshold})的点数: {count} ({count/len(valid)*100:.2f}%)")

print("\n" + "=" * 60)
print("各级质控结果统计")
print("=" * 60)

qc_df = init_qc_codes(df)

qc1 = qc_level1_range(df, qc_df)
print("\n第1级-范围检查后:")
for param in ['wind_speed', 'air_temp', 'pressure', 'Hs', 'SST', 'salinity']:
    if param in qc1.columns:
        counts = qc1[param].value_counts().to_dict()
        total = len(qc1)
        print(f"  {param}: 错误={counts.get(QC_ERROR,0)}({counts.get(QC_ERROR,0)/total*100:.1f}%), "
              f"缺测={counts.get(QC_MISSING,0)}({counts.get(QC_MISSING,0)/total*100:.1f}%), "
              f"未检={counts.get(QC_UNCHECKED,0)}({counts.get(QC_UNCHECKED,0)/total*100:.1f}%)")

qc2 = qc_level2_temporal(df, qc1)
print("\n第2级-时间一致性后:")
for param in ['wind_speed', 'air_temp', 'pressure', 'Hs', 'SST', 'salinity']:
    if param in qc2.columns:
        counts = qc2[param].value_counts().to_dict()
        total = len(qc2)
        print(f"  {param}: 可疑={counts.get(QC_SUSPECT,0)}({counts.get(QC_SUSPECT,0)/total*100:.1f}%), "
              f"错误={counts.get(QC_ERROR,0)}({counts.get(QC_ERROR,0)/total*100:.1f}%), "
              f"缺测={counts.get(QC_MISSING,0)}")

qc3 = qc_level3_internal(df, qc2)
print("\n第3级-内部一致性后:")
for param in ['wind_speed', 'Hs']:
    if param in qc3.columns:
        counts = qc3[param].value_counts().to_dict()
        total = len(qc3)
        print(f"  {param}: 可疑={counts.get(QC_SUSPECT,0)}({counts.get(QC_SUSPECT,0)/total*100:.1f}%)")

qc4 = qc_level4_climatology(df, qc3)
print("\n第4级-气候学检查后:")
for param in ['wind_speed', 'air_temp', 'pressure', 'Hs', 'SST', 'salinity']:
    if param in qc4.columns:
        counts = qc4[param].value_counts().to_dict()
        total = len(qc4)
        print(f"  {param}: 可疑={counts.get(QC_SUSPECT,0)}({counts.get(QC_SUSPECT,0)/total*100:.1f}%), "
              f"错误={counts.get(QC_ERROR,0)}")

qc5 = qc_level5_spike(df, qc4)
print("\n第5级-尖峰检测后:")
for param in ['wind_speed', 'air_temp', 'pressure', 'Hs', 'SST', 'salinity']:
    if param in qc5.columns:
        counts = qc5[param].value_counts().to_dict()
        total = len(qc5)
        print(f"  {param}: 错误={counts.get(QC_ERROR,0)}({counts.get(QC_ERROR,0)/total*100:.1f}%), "
              f"可疑={counts.get(QC_SUSPECT,0)}")

qc6 = qc_level6_stuck(df, qc5)
print("\n第6级-卡值检测后:")
for param in ['wind_speed', 'air_temp', 'pressure', 'Hs', 'SST', 'salinity']:
    if param in qc6.columns:
        counts = qc6[param].value_counts().to_dict()
        total = len(qc6)
        print(f"  {param}: 可疑={counts.get(QC_SUSPECT,0)}({counts.get(QC_SUSPECT,0)/total*100:.1f}%)")

final = run_full_qc(df)
print("\n" + "=" * 60)
print("最终质控结果")
print("=" * 60)
for param in ['wind_speed', 'air_temp', 'pressure', 'Hs', 'SST', 'salinity']:
    if param in final.qc_codes.columns:
        counts = final.qc_codes[param].value_counts().to_dict()
        total = len(final.qc_codes)
        good = counts.get(QC_GOOD, 0)
        suspect = counts.get(QC_SUSPECT, 0)
        error = counts.get(QC_ERROR, 0)
        missing = counts.get(QC_MISSING, 0)
        print(f"  {param}:")
        print(f"    合格={good} ({good/total*100:.1f}%)")
        print(f"    可疑={suspect} ({suspect/total*100:.1f}%)")
        print(f"    错误={error} ({error/total*100:.1f}%)")
        print(f"    缺测={missing} ({missing/total*100:.1f}%)")
