import pandas as pd
import numpy as np
from scipy import interpolate
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from io import BytesIO

from .quality_control import QC_MISSING, QC_INTERPOLATED, QC_GOOD, PARAMETERS


def detect_missing_gaps(series: pd.Series, threshold_hours: float = 3.0) -> List[Dict]:
    valid = series.dropna()
    if len(valid) < 2:
        return []
    gaps = []
    times = valid.index.sort_values()
    for i in range(len(times) - 1):
        gap_hours = (times[i + 1] - times[i]).total_seconds() / 3600.0
        if gap_hours > threshold_hours:
            gaps.append({
                'start': times[i],
                'end': times[i + 1],
                'duration_hours': gap_hours
            })
    return gaps


def interpolate_missing(series: pd.Series, method: str = 'linear',
                        max_gap_hours: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    result = series.copy()
    interpolated_mask = pd.Series(False, index=series.index)
    dt_hours = 1.0
    times = series.index.sort_values()
    if len(times) > 1:
        dt_hours = (times[1] - times[0]).total_seconds() / 3600.0
    max_points = int(max_gap_hours / dt_hours) if dt_hours > 0 else 3
    if method == 'linear':
        for i in range(len(series)):
            if pd.isna(series.iloc[i]):
                left_idx = i - 1
                while left_idx >= 0 and pd.isna(series.iloc[left_idx]):
                    left_idx -= 1
                right_idx = i + 1
                while right_idx < len(series) and pd.isna(series.iloc[right_idx]):
                    right_idx += 1
                if left_idx >= 0 and right_idx < len(series):
                    gap_size = right_idx - left_idx - 1
                    if gap_size <= max_points:
                        left_val = series.iloc[left_idx]
                        right_val = series.iloc[right_idx]
                        fraction = (i - left_idx) / (right_idx - left_idx)
                        result.iloc[i] = left_val + fraction * (right_val - left_val)
                        interpolated_mask.iloc[i] = True
    elif method == 'spline':
        valid = series.dropna()
        if len(valid) >= 4:
            valid_idx = np.arange(len(series))[series.notna()]
            valid_vals = valid.values
            try:
                tck = interpolate.splrep(valid_idx, valid_vals, k=3, s=0)
                all_idx = np.arange(len(series))
                interp_vals = interpolate.splev(all_idx, tck)
                for i in range(len(series)):
                    if pd.isna(series.iloc[i]):
                        left_idx = i - 1
                        while left_idx >= 0 and pd.isna(series.iloc[left_idx]):
                            left_idx -= 1
                        right_idx = i + 1
                        while right_idx < len(series) and pd.isna(series.iloc[right_idx]):
                            right_idx += 1
                        if left_idx >= 0 and right_idx < len(series):
                            gap_size = right_idx - left_idx - 1
                            if gap_size <= max_points:
                                result.iloc[i] = interp_vals[i]
                                interpolated_mask.iloc[i] = True
            except Exception:
                pass
    return result, interpolated_mask


def process_missing_data(df: pd.DataFrame, qc_codes: pd.DataFrame,
                         method: str = 'linear', max_gap_hours: float = 3.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_result = df.copy()
    qc_result = qc_codes.copy()
    gap_info = {}
    for buoy_id in df['buoy_id'].unique():
        buoy_mask = df['buoy_id'] == buoy_id
        buoy_df = df.loc[buoy_mask].sort_values('time').set_index('time')
        buoy_qc = qc_codes.loc[buoy_mask].sort_values('time').set_index('time')
        gap_info[buoy_id] = {}
        for param in PARAMETERS:
            if param not in buoy_df.columns:
                continue
            series = buoy_df[param]
            gaps = detect_missing_gaps(series, max_gap_hours)
            gap_info[buoy_id][param] = gaps
            interpolated, mask = interpolate_missing(series, method, max_gap_hours)
            buoy_df.loc[mask, param] = interpolated[mask]
            buoy_qc.loc[mask, param] = QC_INTERPOLATED
        buoy_df = buoy_df.reset_index()
        buoy_qc = buoy_qc.reset_index()
        df_result.loc[buoy_mask] = buoy_df.set_index(df_result.index[buoy_mask])
        qc_result.loc[buoy_mask] = buoy_qc.set_index(qc_result.index[buoy_mask])
    return df_result, qc_result, gap_info


def generate_pdf_report(output_path: str, buoy_id: str, start_time: str, end_time: str,
                        report_id: str, data_overview: Dict, qc_stats: Dict,
                        wave_figures: List = None, current_figures: List = None,
                        extreme_figures: List = None, meteorology_figures: List = None) -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError:
        raise ImportError("请安装reportlab库以生成PDF报告")
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            rightMargin=2 * cm, leftMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm)
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=24, spaceAfter=30)
    h1_style = ParagraphStyle('Heading1', parent=styles['Heading1'], fontSize=18, spaceAfter=15)
    h2_style = ParagraphStyle('Heading2', parent=styles['Heading2'], fontSize=14, spaceAfter=10)
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=10, leading=14)
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("海况分析报告", title_style))
    story.append(Spacer(1, 2 * cm))
    cover_data = [
        ['报告编号:', report_id],
        ['站点:', buoy_id],
        ['时间范围:', f'{start_time} 至 {end_time}'],
    ]
    cover_table = Table(cover_data, colWidths=[4 * cm, 10 * cm])
    cover_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 12),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(cover_table)
    story.append(PageBreak())
    story.append(Paragraph("一、数据概况", h1_style))
    if buoy_id in data_overview:
        info = data_overview[buoy_id]
        overview_data = [
            ['起始时间', str(info['time_start'])],
            ['结束时间', str(info['time_end'])],
            ['记录总数', str(info['record_count'])],
        ]
        story.append(Table(overview_data, colWidths=[5 * cm, 8 * cm]))
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph("各参数缺测情况:", h2_style))
        missing_data = [['参数', '总记录数', '缺测数', '缺测比例(%)']]
        for param, stats in info['parameters'].items():
            missing_data.append([param, str(stats['total']), str(stats['missing']),
                                 f"{stats['missing_rate'] * 100:.2f}"])
        missing_table = Table(missing_data, colWidths=[3 * cm, 3 * cm, 3 * cm, 3 * cm])
        missing_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ]))
        story.append(missing_table)
    story.append(PageBreak())
    story.append(Paragraph("二、质控统计", h1_style))
    if 7 in qc_stats:
        final_stats = qc_stats[7]
        qc_labels = {0: '未检', 1: '合格', 2: '可疑', 3: '错误', 4: '缺测', 5: '插补'}
        qc_summary = [['参数', '合格', '可疑', '错误', '缺测', '插补']]
        for param in PARAMETERS:
            if param in final_stats:
                counts = final_stats[param]
                qc_summary.append([param,
                                   str(counts.get(1, 0)), str(counts.get(2, 0)),
                                   str(counts.get(3, 0)), str(counts.get(4, 0)),
                                   str(counts.get(5, 0))])
        qc_table = Table(qc_summary, colWidths=[2.5 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm])
        qc_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
        ]))
        story.append(qc_table)
    story.append(PageBreak())
    story.append(Paragraph("三、波浪分析", h1_style))
    if wave_figures:
        for fig in wave_figures:
            if isinstance(fig, Figure):
                img_data = BytesIO()
                fig.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
                img_data.seek(0)
                img = Image(img_data, width=15 * cm, height=10 * cm)
                story.append(img)
                story.append(Spacer(1, 0.5 * cm))
    story.append(PageBreak())
    story.append(Paragraph("四、海流分析", h1_style))
    if current_figures:
        for fig in current_figures:
            if isinstance(fig, Figure):
                img_data = BytesIO()
                fig.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
                img_data.seek(0)
                img = Image(img_data, width=15 * cm, height=10 * cm)
                story.append(img)
                story.append(Spacer(1, 0.5 * cm))
    story.append(PageBreak())
    story.append(Paragraph("五、极值统计", h1_style))
    if extreme_figures:
        for fig in extreme_figures:
            if isinstance(fig, Figure):
                img_data = BytesIO()
                fig.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
                img_data.seek(0)
                img = Image(img_data, width=15 * cm, height=10 * cm)
                story.append(img)
                story.append(Spacer(1, 0.5 * cm))
    story.append(PageBreak())
    story.append(Paragraph("六、气象玫瑰图", h1_style))
    if meteorology_figures:
        for fig in meteorology_figures:
            if isinstance(fig, Figure):
                img_data = BytesIO()
                fig.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
                img_data.seek(0)
                img = Image(img_data, width=12 * cm, height=12 * cm)
                story.append(img)
                story.append(Spacer(1, 0.5 * cm))
    doc.build(story)
