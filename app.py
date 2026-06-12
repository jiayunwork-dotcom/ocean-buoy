import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import sys
import os
import json
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.plot_utils import setup_chinese_font
setup_chinese_font()

from src.data_loader import (
    load_file, merge_buoy_data, get_data_overview, CSV_COLUMNS
)
from src.quality_control import (
    run_full_qc, QCResult, PARAMETERS, QC_GOOD, QC_SUSPECT, QC_ERROR,
    QC_MISSING, QC_INTERPOLATED, DEFAULT_RANGE_THRESHOLDS, DEFAULT_GRADIENT_THRESHOLDS
)
from src.qc_visualization import (
    plot_qc_timeseries, plot_qc_statistics, plot_qc_level_stats,
    QC_LABELS, apply_manual_override, apply_batch_override,
    extract_error_events, PARAM_NAMES_CN, QC_LEVEL_NAMES
)
from src.wave_analysis import analyze_wave_buoy, compute_wave_monthly_stats, plot_wave_monthly_bar
from src.current_analysis import analyze_current_buoy
from src.extreme_stats import analyze_extremes, RETURN_PERIODS
from src.meteorology_analysis import (
    analyze_meteorology_buoy, plot_multi_buoy_comparison,
    compute_monthly_means_table, compute_correlation_matrix,
    plot_param_correlation_heatmap
)
from src.report_generator import process_missing_data, generate_pdf_report
from src.warning_engine import (
    WARNING_LEVELS, OPERATORS, PARAM_NAMES_CN, AVAILABLE_PARAMS,
    Condition, WarningRule, WarningEvent,
    get_builtin_templates, make_unique_name, import_templates,
    scan_warnings, events_to_dataframe,
    compute_level_counts, compute_buoy_counts, compute_hourly_heatmap,
    serialize_rules, deserialize_rules,
    apply_escalation, build_composite_groups,
    detect_cycle_dependency, get_prerequisite_chain,
    compute_daily_trend, compute_daily_totals,
    compute_7day_moving_avg, detect_anomaly_days,
    topo_sort_rules
)

st.set_page_config(page_title="海洋浮标数据质控与海况分析系统", layout="wide")
st.title("🌊 海洋浮标观测数据质量控制与海况分析系统")


if 'data' not in st.session_state:
    st.session_state.data = None
if 'qc_result' not in st.session_state:
    st.session_state.qc_result = None
if 'data_overview' not in st.session_state:
    st.session_state.data_overview = None
if 'warning_rules' not in st.session_state:
    st.session_state.warning_rules = []
if 'warning_events' not in st.session_state:
    st.session_state.warning_events = None
if 'warning_events_df' not in st.session_state:
    st.session_state.warning_events_df = None
if 'warning_composite_groups' not in st.session_state:
    st.session_state.warning_composite_groups = []
if 'warning_escalation_state' not in st.session_state:
    st.session_state.warning_escalation_state = {}


st.sidebar.header("导航")
page = st.sidebar.radio("功能模块", [
    "📥 数据导入",
    "🔍 质量控制",
    "🌊 波浪分析",
    "🌊 海流分析",
    "📊 极值统计",
    "🌤️ 气象分析",
    "📊 多浮标对比",
    "🔧 缺测处理",
    "🚨 预警管理",
    "📄 报告生成"
])

if page == "📥 数据导入":
    st.header("数据导入")
    st.markdown("支持CSV和NetCDF格式的浮标观测数据导入，可同时导入多个浮标数据文件。")
    uploaded_files = st.file_uploader("上传浮标数据文件", type=['csv', 'nc'], accept_multiple_files=True)
    if uploaded_files:
        dataframes = []
        for file in uploaded_files:
            try:
                df = load_file(file)
                dataframes.append(df)
                st.success(f"成功导入: {file.name} ({len(df)} 条记录)")
            except Exception as e:
                st.error(f"导入失败 {file.name}: {str(e)}")
        if dataframes:
            merged_data = merge_buoy_data(dataframes)
            st.session_state.data = merged_data
            st.session_state.data_overview = get_data_overview(merged_data)
            st.session_state.qc_result = None
            st.success(f"✅ 数据合并完成，共 {len(merged_data)} 条记录")
    if st.session_state.data is not None:
        st.subheader("数据概况")
        overview = st.session_state.data_overview
        for buoy_id, info in overview.items():
            with st.expander(f"🚩 {buoy_id}"):
                col1, col2, col3 = st.columns(3)
                col1.info(f"**起始时间**: {info['time_start']}")
                col2.info(f"**结束时间**: {info['time_end']}")
                col3.info(f"**记录总数**: {info['record_count']}")
                st.markdown("**各参数缺测比例:**")
                missing_df = pd.DataFrame({
                    '参数': list(info['parameters'].keys()),
                    '总记录': [v['total'] for v in info['parameters'].values()],
                    '缺测数': [v['missing'] for v in info['parameters'].values()],
                    '缺测比例(%)': [f"{v['missing_rate'] * 100:.2f}" for v in info['parameters'].values()]
                })
                st.dataframe(missing_df, use_container_width=True)
        with st.expander("🔍 查看原始数据"):
            st.dataframe(st.session_state.data, use_container_width=True)

elif page == "🔍 质量控制":
    st.header("7级质量控制")
    if st.session_state.data is None:
        st.warning("请先在【数据导入】模块上传数据")
    else:
        st.subheader("质控参数设置")
        col_imp_exp1, col_imp_exp2 = st.columns(2)
        with col_imp_exp1:
            uploaded_config = st.file_uploader("📥 导入质控配置(JSON)", type=['json'], key="qc_config_import")
            if uploaded_config is not None:
                try:
                    config_data = json.load(uploaded_config)
                    required_keys = ['range_thresholds', 'gradient_thresholds', 'spike_sigma', 'stuck_n']
                    if not all(k in config_data for k in required_keys):
                        st.error("❌ JSON格式错误：缺少必要键(range_thresholds, gradient_thresholds, spike_sigma, stuck_n)")
                    else:
                        st.session_state.imported_config = config_data
                        for param, thresh in config_data.get('range_thresholds', {}).items():
                            if f"range_min_{param}" in st.session_state:
                                st.session_state[f"range_min_{param}"] = float(thresh.get('min', DEFAULT_RANGE_THRESHOLDS.get(param, {}).get('min', 0)))
                            if f"range_max_{param}" in st.session_state:
                                st.session_state[f"range_max_{param}"] = float(thresh.get('max', DEFAULT_RANGE_THRESHOLDS.get(param, {}).get('max', 100)))
                        for param, val in config_data.get('gradient_thresholds', {}).items():
                            if f"grad_{param}" in st.session_state:
                                st.session_state[f"grad_{param}"] = float(val)
                        if 'spike_sigma_input' in st.session_state:
                            st.session_state['spike_sigma_input'] = float(config_data['spike_sigma'])
                        if 'stuck_n_input' in st.session_state:
                            st.session_state['stuck_n_input'] = int(config_data['stuck_n'])
                        st.success("✅ 配置导入成功！参数已更新")
                        st.rerun()
                except json.JSONDecodeError as e:
                    st.error(f"❌ JSON解析失败：{str(e)}")
                except Exception as e:
                    st.error(f"❌ 导入失败：{str(e)}")
        with col_imp_exp2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📤 导出当前质控配置(JSON)"):
                exp_range = {}
                for param, thresh in DEFAULT_RANGE_THRESHOLDS.items():
                    exp_range[param] = {
                        'min': st.session_state.get(f"range_min_{param}", thresh['min']),
                        'max': st.session_state.get(f"range_max_{param}", thresh['max'])
                    }
                exp_grad = {}
                for param, thresh in DEFAULT_GRADIENT_THRESHOLDS.items():
                    exp_grad[param] = st.session_state.get(f"grad_{param}", thresh)
                export_config = {
                    'range_thresholds': exp_range,
                    'gradient_thresholds': exp_grad,
                    'spike_sigma': st.session_state.get('spike_sigma_input', 3.0),
                    'stuck_n': int(st.session_state.get('stuck_n_input', 6))
                }
                config_json = json.dumps(export_config, ensure_ascii=False, indent=2)
                st.download_button(
                    label="💾 下载配置文件",
                    data=config_json,
                    file_name="qc_config.json",
                    mime="application/json"
                )
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("📏 范围检查阈值 (第1级)"):
                range_thresholds = {}
                for param, thresh in DEFAULT_RANGE_THRESHOLDS.items():
                    c1, c2 = st.columns(2)
                    range_thresholds[param] = {
                        'min': c1.number_input(f"{param} 下限", value=thresh['min'], key=f"range_min_{param}"),
                        'max': c2.number_input(f"{param} 上限", value=thresh['max'], key=f"range_max_{param}")
                    }
        with col2:
            with st.expander("📈 时间一致性阈值 (第2级)"):
                gradient_thresholds = {}
                for param, thresh in DEFAULT_GRADIENT_THRESHOLDS.items():
                    gradient_thresholds[param] = st.number_input(
                        f"{param} 1小时最大变化", value=thresh, key=f"grad_{param}"
                    )
        st.subheader("高级质控参数")
        col3, col4, col5 = st.columns(3)
        with col3:
            spike_sigma = st.number_input("尖峰检测σ阈值 (第5级)", 
                                           value=3.0, 
                                           min_value=1.0, max_value=10.0, key="spike_sigma_input")
        with col4:
            stuck_n = st.number_input("卡值检测连续点数 (第6级)", 
                                       value=6, 
                                       min_value=3, max_value=20, key="stuck_n_input")
        with col5:
            st.info("空间一致性检查(第7级): 需至少2个浮标")
        if st.button("▶️ 运行质控", type="primary"):
            with st.spinner("正在执行7级质控..."):
                qc_result = run_full_qc(
                    st.session_state.data,
                    range_thresholds=range_thresholds,
                    gradient_thresholds=gradient_thresholds,
                    spike_sigma=spike_sigma,
                    stuck_n=stuck_n
                )
                st.session_state.qc_result = qc_result
                st.success("✅ 质控完成！")
        if st.session_state.qc_result is not None:
            qc_result = st.session_state.qc_result
            tab_qc1, tab_qc2, tab_qc3 = st.tabs(["质控统计", "结果可视化", "⚠️ 异常事件"])
            with tab_qc1:
                st.subheader("质控统计")
                t1, t2 = st.tabs(["各参数质控分布", "各级质控演变"])
                with t1:
                    st.pyplot(plot_qc_statistics(qc_result.qc_codes))
                with t2:
                    st.pyplot(plot_qc_level_stats(qc_result.level_stats))
            with tab_qc2:
                st.subheader("质控结果可视化")
                buoy_ids = st.session_state.data['buoy_id'].unique()
                col_buoy, col_param = st.columns(2)
                selected_buoy = col_buoy.selectbox("选择浮标", buoy_ids)
                param_names_cn = {
                    'wind_speed': '风速', 'wind_dir': '风向', 'air_temp': '气温',
                    'pressure': '气压', 'Hs': '有效波高', 'Tz': '平均波周期',
                    'wave_dir': '波向', 'SST': '海表温度', 'salinity': '盐度',
                    'current_speed': '流速', 'current_dir': '流向'
                }
                selected_param = col_param.selectbox("选择参数", PARAMETERS,
                                                      format_func=lambda x: f"{param_names_cn[x]} ({x})")
                st.pyplot(plot_qc_timeseries(
                    st.session_state.data, qc_result.qc_codes,
                    selected_buoy, selected_param, qc_result.manual_overrides
                ))
                st.subheader("人工修正")
                with st.expander("✏️ 单点修正"):
                    buoy_df = st.session_state.data[st.session_state.data['buoy_id'] == selected_buoy].sort_values('time')
                    time_options = buoy_df['time'].tolist()
                    selected_time = st.selectbox("选择时间点", time_options,
                                                 format_func=lambda x: x.strftime("%Y-%m-%d %H:%M"))
                    new_qc_code = st.selectbox("新质控码", [QC_GOOD, QC_SUSPECT, QC_ERROR, QC_MISSING],
                                                format_func=lambda x: QC_LABELS[x])
                    reason = st.text_input("修改原因")
                    if st.button("应用单点修改"):
                        st.session_state.qc_result = apply_manual_override(
                            qc_result, selected_buoy, selected_time,
                            selected_param, new_qc_code, reason
                        )
                        st.success("修改已应用")
                        st.rerun()
                with st.expander("📋 批量修正"):
                    start_idx = st.selectbox("起始时间索引", range(len(time_options)),
                                              format_func=lambda i: time_options[i].strftime("%Y-%m-%d %H:%M"))
                    end_idx = st.selectbox("结束时间索引", range(len(time_options)),
                                            index=min(start_idx + 100, len(time_options) - 1),
                                            format_func=lambda i: time_options[i].strftime("%Y-%m-%d %H:%M"))
                    batch_qc_code = st.selectbox("批量设置质控码", [QC_GOOD, QC_SUSPECT, QC_ERROR, QC_MISSING],
                                                  format_func=lambda x: QC_LABELS[x], key="batch_code")
                    batch_reason = st.text_input("批量修改原因", key="batch_reason")
                    if st.button("应用批量修改"):
                        st.session_state.qc_result = apply_batch_override(
                            qc_result, selected_buoy,
                            time_options[start_idx], time_options[end_idx],
                            selected_param, batch_qc_code, batch_reason
                        )
                        st.success(f"已修改 {end_idx - start_idx + 1} 个数据点")
                        st.rerun()
            with tab_qc3:
                st.subheader("⚠️ 异常事件时间线")
                events_df = extract_error_events(st.session_state.data, qc_result)
                if len(events_df) == 0:
                    st.info("暂无异常事件")
                else:
                    st.info(f"共检测到 {len(events_df)} 个异常事件")
                    col_f1, col_f2 = st.columns(2)
                    all_buoys = ['全部'] + list(events_df['buoy_id'].unique())
                    filter_buoy = col_f1.selectbox("按浮标筛选", all_buoys)
                    all_levels = ['全部'] + sorted([l for l in events_df['qc_level'].dropna().unique() if l > 0])
                    all_level_labels = ['全部'] + [f"第{int(l)}级 - {QC_LEVEL_NAMES.get(int(l), '')}" for l in all_levels if l != '全部']
                    filter_level_idx = col_f2.selectbox("按质控级别筛选", range(len(all_levels)),
                                                         format_func=lambda i: all_level_labels[i])
                    filtered = events_df.copy()
                    if filter_buoy != '全部':
                        filtered = filtered[filtered['buoy_id'] == filter_buoy]
                    if filter_level_idx > 0:
                        lv = all_levels[filter_level_idx]
                        filtered = filtered[filtered['qc_level'] == lv]
                    if len(filtered) == 0:
                        st.warning("筛选条件下无事件")
                    else:
                        page_size = 20
                        total_pages = (len(filtered) - 1) // page_size + 1
                        col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
                        current_page = col_p2.slider("页码", 1, total_pages, 1)
                        start_idx = (current_page - 1) * page_size
                        end_idx = min(start_idx + page_size, len(filtered))
                        page_events = filtered.iloc[start_idx:end_idx]
                        st.markdown(f"第 {current_page}/{total_pages} 页，显示 {start_idx+1}-{end_idx} 条")
                        for _, ev in page_events.iterrows():
                            level_badge = ""
                            if pd.notna(ev['qc_level']):
                                lv = int(ev['qc_level'])
                                level_badge = f'<span style="background:#ff4b4b;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">第{lv}级 · {QC_LEVEL_NAMES.get(lv, "")}</span>'
                            value_str = f"{ev['original_value']:.4f}" if pd.notna(ev['original_value']) and isinstance(ev['original_value'], (int, float)) else str(ev['original_value'])
                            card_html = f"""
                            <div style="border:1px solid #e0e0e0;border-radius:8px;padding:12px;margin-bottom:10px;background:#fafafa;">
                                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                                    <span style="font-weight:bold;font-size:15px;">⏰ {ev['time'].strftime('%Y-%m-%d %H:%M:%S')}</span>
                                    {level_badge}
                                </div>
                                <div style="display:flex;gap:20px;font-size:14px;">
                                    <span>🚩 <b>{ev['buoy_id']}</b></span>
                                    <span>📊 <b>{ev['param_name']}</b> ({ev['param']})</span>
                                    <span>📈 原始值: <b>{value_str}</b></span>
                                </div>
                            </div>
                            """
                            st.markdown(card_html, unsafe_allow_html=True)

elif page == "🌊 波浪分析":
    st.header("波浪分析")
    if st.session_state.data is None:
        st.warning("请先在【数据导入】模块上传数据")
    elif st.session_state.qc_result is None:
        st.warning("请先在【质量控制】模块运行质控")
    else:
        buoy_ids = st.session_state.data['buoy_id'].unique()
        selected_buoy = st.selectbox("选择浮标", buoy_ids)
        tab_w1, tab_w2 = st.tabs(["🌊 波浪谱分析", "📊 月度统计"])
        with tab_w1:
            if st.button("▶️ 分析波浪数据", type="primary"):
                with st.spinner("正在分析波浪数据..."):
                    wave_result = analyze_wave_buoy(
                        st.session_state.data, selected_buoy,
                        st.session_state.qc_result.qc_codes
                    )
                    if 'error' in wave_result:
                        st.warning(f"⚠️ {wave_result['error']}")
                    else:
                        st.subheader("波浪谱参数")
                        params = wave_result['params']
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("峰值频率 fp", f"{params['fp']:.4f} Hz")
                        col2.metric("峰值周期 Tp", f"{params['Tp']:.2f} s")
                        col3.metric("有效波高 Hm0", f"{params['Hm0']:.3f} m")
                        col4.metric("平均跨零周期 Tz", f"{params['Tz']:.2f} s")
                        col5, col6, col7, col8 = st.columns(4)
                        col5.metric("谱矩 m0", f"{params['m0']:.6f}")
                        col6.metric("谱矩 m1", f"{params['m1']:.6f}")
                        col7.metric("谱矩 m2", f"{params['m2']:.6f}")
                        col8.metric("谱宽度 ε", f"{params['epsilon']:.4f}")
                        st.subheader("波浪谱密度曲线")
                        st.pyplot(wave_result['spectrum_fig'])
                        if 'dir_spectrum_fig' in wave_result:
                            st.subheader("波浪方向谱")
                            st.pyplot(wave_result['dir_spectrum_fig'])
        with tab_w2:
            st.subheader("波高月度统计")
            if st.button("▶️ 生成月度统计", type="primary", key="wave_monthly"):
                with st.spinner("正在计算月度统计..."):
                    monthly_stats = compute_wave_monthly_stats(
                        st.session_state.data, selected_buoy,
                        st.session_state.qc_result.qc_codes
                    )
                    if len(monthly_stats) == 0:
                        st.warning("无有效波高数据")
                    else:
                        st.dataframe(
                            monthly_stats.style.format({
                                '平均波高(m)': '{:.3f}',
                                '最大波高(m)': '{:.3f}',
                                '波高标准差(m)': '{:.3f}',
                                '超过2m天数占比': '{:.2%}'
                            }),
                            use_container_width=True
                        )
                        csv = monthly_stats.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 下载CSV",
                            data=csv,
                            file_name=f"波浪月度统计_{selected_buoy}.csv",
                            mime="text/csv"
                        )
                        st.subheader("月度平均波高柱状图")
                        fig = plot_wave_monthly_bar(monthly_stats)
                        st.plotly_chart(fig, use_container_width=True)

elif page == "🌊 海流分析":
    st.header("潮流调和与余流分析")
    if st.session_state.data is None:
        st.warning("请先在【数据导入】模块上传数据")
    elif st.session_state.qc_result is None:
        st.warning("请先在【质量控制】模块运行质控")
    else:
        buoy_ids = st.session_state.data['buoy_id'].unique()
        selected_buoy = st.selectbox("选择浮标", buoy_ids)
        if st.button("▶️ 分析海流数据", type="primary"):
            with st.spinner("正在分析海流数据..."):
                current_result = analyze_current_buoy(
                    st.session_state.data, selected_buoy,
                    st.session_state.qc_result.qc_codes
                )
                if 'tidal_warning' in current_result:
                    st.warning(f"⚠️ {current_result['tidal_warning']}")
                if 'tidal' in current_result:
                    st.subheader("潮流调和分析结果")
                    tidal = current_result['tidal']
                    ellipses = tidal['ellipses']
                    ellipse_df = pd.DataFrame([
                        {
                            '分潮': name,
                            '长半轴(m/s)': f"{p['semi_major']:.4f}",
                            '短半轴(m/s)': f"{p['semi_minor']:.4f}",
                            '倾角(°)': f"{p['inclination']:.2f}",
                            '偏心率': f"{p['eccentricity']:.4f}"
                        } for name, p in ellipses.items()
                    ])
                    st.dataframe(ellipse_df, use_container_width=True)
                    st.subheader("潮流椭圆")
                    st.pyplot(current_result['tidal_ellipses_fig'])
                if 'residual_fig' in current_result:
                    st.subheader("余流分析")
                    st.pyplot(current_result['residual_fig'])
                if 'current_rose_fig' in current_result:
                    st.subheader("流速流向玫瑰图")
                    st.pyplot(current_result['current_rose_fig'])

elif page == "📊 极值统计":
    st.header("极值统计与重现期分析")
    if st.session_state.data is None:
        st.warning("请先在【数据导入】模块上传数据")
    elif st.session_state.qc_result is None:
        st.warning("请先在【质量控制】模块运行质控")
    else:
        buoy_ids = st.session_state.data['buoy_id'].unique()
        selected_buoy = st.selectbox("选择浮标", buoy_ids)
        if st.button("▶️ 分析极值统计", type="primary"):
            with st.spinner("正在进行极值统计分析..."):
                extreme_result = analyze_extremes(
                    st.session_state.data, selected_buoy,
                    st.session_state.qc_result.qc_codes
                )
                if 'warning' in extreme_result:
                    st.warning(f"⚠️ {extreme_result['warning']}")
                    if 'histogram_fig' in extreme_result:
                        st.pyplot(extreme_result['histogram_fig'])
                else:
                    st.subheader("年最大有效波高序列")
                    st.dataframe(
                        pd.DataFrame({
                            '年份': extreme_result['annual_max'].index.year,
                            '最大波高(m)': extreme_result['annual_max'].values
                        }),
                        use_container_width=True
                    )
                    st.subheader("分布拟合结果")
                    if 'fits' in extreme_result:
                        for fit in extreme_result['fits']:
                            with st.expander(f"📐 {fit['distribution']} 分布"):
                                col1, col2, col3 = st.columns(3)
                                col1.metric("对数似然值", f"{fit['log_likelihood']:.2f}")
                                col2.metric("AIC", f"{fit['aic']:.2f}")
                                if 'ks_test' in fit:
                                    col3.metric("K-S检验 p值", f"{fit['ks_test']['p_value']:.4f}")
                                params_df = pd.DataFrame([
                                    {'参数': k, '值': f"{v:.4f}"} for k, v in fit['params'].items()
                                ])
                                st.dataframe(params_df, use_container_width=True)
                    st.subheader("重现期-波高关系曲线")
                    if 'return_curve_fig' in extreme_result:
                        st.pyplot(extreme_result['return_curve_fig'])
                    st.subheader("设计波高表")
                    if 'return_levels' in extreme_result:
                        rl_data = []
                        for dist, rps in extreme_result['return_levels'].items():
                            for rp, info in rps.items():
                                rl_data.append({
                                    '分布': dist,
                                    '重现期(年)': rp,
                                    '设计波高(m)': f"{info['value']:.3f}",
                                    '95%CI下限': f"{info['ci_lower']:.3f}",
                                    '95%CI上限': f"{info['ci_upper']:.3f}"
                                })
                        st.dataframe(pd.DataFrame(rl_data), use_container_width=True)
                    st.subheader("Q-Q图")
                    if 'qq_figs' in extreme_result:
                        for dist, fig in extreme_result['qq_figs'].items():
                            st.markdown(f"**{dist} 分布**")
                            st.pyplot(fig)

elif page == "🌤️ 气象分析":
    st.header("气象要素分析")
    if st.session_state.data is None:
        st.warning("请先在【数据导入】模块上传数据")
    elif st.session_state.qc_result is None:
        st.warning("请先在【质量控制】模块运行质控")
    else:
        buoy_ids = st.session_state.data['buoy_id'].unique()
        selected_buoy = st.selectbox("选择浮标", buoy_ids)
        if st.button("▶️ 分析气象数据", type="primary"):
            with st.spinner("正在分析气象数据..."):
                met_result = analyze_meteorology_buoy(
                    st.session_state.data, selected_buoy,
                    st.session_state.qc_result.qc_codes
                )
                if 'wind_rose' in met_result:
                    st.subheader("风速风向玫瑰图")
                    st.pyplot(met_result['wind_rose'])
                if 'pressure_plot' in met_result:
                    st.subheader("气压时序趋势")
                    st.pyplot(met_result['pressure_plot'])
                    if met_result.get('low_pressure_events'):
                        st.markdown("**检测到的低压过境事件:**")
                        for event in met_result['low_pressure_events']:
                            st.info(f"⏰ {event['time'].strftime('%Y-%m-%d %H:%M')} "
                                    f"| 6h降压: {event['pressure_drop']:.1f} hPa")
                if 'temp_diurnal' in met_result:
                    st.subheader("气温日变化")
                    st.pyplot(met_result['temp_diurnal'])

elif page == "📊 多浮标对比":
    st.header("多浮标对比分析")
    if st.session_state.data is None:
        st.warning("请先在【数据导入】模块上传数据")
    else:
        buoy_ids = st.session_state.data['buoy_id'].unique()
        tab_m1, tab_m2 = st.tabs(["📊 多浮标对比", "🔥 参数相关性"])
        with tab_m1:
            if len(buoy_ids) < 2:
                st.warning("需要至少2个浮标数据才能进行对比分析")
            else:
                selected_buoys = st.multiselect("选择浮标 (2-4个)", buoy_ids,
                                                 default=list(buoy_ids)[:min(4, len(buoy_ids))])
                param_names_cn = {
                    'wind_speed': '风速', 'wind_dir': '风向', 'air_temp': '气温',
                    'pressure': '气压', 'Hs': '有效波高', 'Tz': '平均波周期',
                    'SST': '海表温度', 'salinity': '盐度'
                }
                selected_param = st.selectbox("选择对比参数",
                                              [p for p in PARAMETERS if p != 'wave_dir' and p != 'current_dir'],
                                              format_func=lambda x: f"{param_names_cn.get(x, x)} ({x})")
                if len(selected_buoys) >= 2 and st.button("▶️ 生成对比分析", type="primary"):
                    st.subheader("时序对比")
                    st.pyplot(plot_multi_buoy_comparison(
                        st.session_state.data, selected_buoys, selected_param
                    ))
                    st.subheader("月均值对比表")
                    monthly_table = compute_monthly_means_table(
                        st.session_state.data, selected_buoys, selected_param
                    )
                    st.dataframe(monthly_table.style.format("{:.3f}"), use_container_width=True)
                    st.subheader("相关系数矩阵")
                    corr_matrix = compute_correlation_matrix(
                        st.session_state.data, selected_buoys, selected_param
                    )
                    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm').format("{:.3f}"),
                                 use_container_width=True)
        with tab_m2:
            st.subheader("单浮标多参数相关性热力图")
            heatmap_buoy = st.selectbox("选择浮标", buoy_ids, key="heatmap_buoy")
            if st.button("▶️ 计算参数相关性", type="primary", key="heatmap_btn"):
                with st.spinner("正在计算相关性矩阵..."):
                    qc_codes = st.session_state.qc_result.qc_codes if st.session_state.qc_result else None
                    fig = plot_param_correlation_heatmap(
                        st.session_state.data, heatmap_buoy, qc_codes
                    )
                    st.plotly_chart(fig, use_container_width=True)

elif page == "🔧 缺测处理":
    st.header("缺测数据处理")
    if st.session_state.data is None:
        st.warning("请先在【数据导入】模块上传数据")
    else:
        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox("插补方法", ["linear", "spline"],
                                   format_func=lambda x: "线性插补" if x == "linear" else "三次样条插补")
        with col2:
            max_gap_hours = st.number_input("最大插补间隔(小时)", value=3.0, min_value=0.5, max_value=24.0)
        if st.button("▶️ 执行缺测处理", type="primary"):
            with st.spinner("正在处理缺测数据..."):
                qc_codes = st.session_state.qc_result.qc_codes if st.session_state.qc_result else None
                if qc_codes is None:
                    from src.quality_control import init_qc_codes
                    qc_codes = init_qc_codes(st.session_state.data)
                df_processed, qc_processed, gap_info = process_missing_data(
                    st.session_state.data, qc_codes, method, max_gap_hours
                )
                st.session_state.data = df_processed
                if st.session_state.qc_result:
                    st.session_state.qc_result.qc_codes = qc_processed
                st.success("✅ 缺测处理完成")
                st.subheader("长时缺测段 (>3小时)")
                for buoy_id, params in gap_info.items():
                    with st.expander(f"🚩 {buoy_id}"):
                        for param, gaps in params.items():
                            if gaps:
                                st.markdown(f"**{param}:**")
                                for gap in gaps[:10]:
                                    st.info(f"  {gap['start'].strftime('%Y-%m-%d %H:%M')} ~ "
                                            f"{gap['end'].strftime('%Y-%m-%d %H:%M')} "
                                            f"({gap['duration_hours']:.1f}小时)")
                                if len(gaps) > 10:
                                    st.info(f"  ... 另有 {len(gaps) - 10} 个缺测段")

elif page == "🚨 预警管理":
    st.header("海况预警与事件管理")
    tab_wr, tab_we, tab_ws = st.tabs(["⚙️ 预警规则管理", "📋 预警事件列表", "📊 预警统计仪表盘"])

    with tab_wr:
        st.subheader("预警规则列表")
        col_imp1, col_imp2, col_imp3 = st.columns(3)
        with col_imp1:
            if st.button("📥 导入内置模板", type="primary"):
                st.session_state.warning_rules = import_templates(st.session_state.warning_rules)
                st.success("✅ 已导入3条内置模板规则")
                st.rerun()
        with col_imp2:
            rule_export = serialize_rules(st.session_state.warning_rules)
            st.download_button(
                label="📤 导出规则(JSON)",
                data=rule_export,
                file_name="warning_rules.json",
                mime="application/json"
            )
        with col_imp3:
            uploaded_rules = st.file_uploader("📁 导入规则(JSON)", type=['json'], key="rule_import", label_visibility="collapsed")
            if uploaded_rules is not None:
                try:
                    json_str = uploaded_rules.read().decode('utf-8')
                    imported = deserialize_rules(json_str)
                    if imported:
                        existing_names = [r.name for r in st.session_state.warning_rules]
                        for r in imported:
                            r.name = make_unique_name(r.name, existing_names)
                            existing_names.append(r.name)
                        st.session_state.warning_rules = st.session_state.warning_rules + imported
                        st.success(f"✅ 成功导入 {len(imported)} 条规则")
                        st.rerun()
                except Exception as e:
                    st.error(f"导入失败: {str(e)}")

        rules = st.session_state.warning_rules
        if len(rules) == 0:
            st.info("暂无预警规则，请添加规则或导入内置模板")
        else:
            rule_rows = []
            for idx, r in enumerate(rules):
                level_info = WARNING_LEVELS.get(r.level, {})
                dep_str = ""
                if r.prerequisite_rule:
                    dep_str = f"← {r.prerequisite_rule}"
                rule_rows.append({
                    '序号': idx + 1,
                    '规则名称': r.name,
                    '预警级别': level_info.get('name', f'L{r.level}'),
                    '触发条件': r.describe_conditions(),
                    '前置规则': dep_str if dep_str else "无",
                    '状态': '✅ 启用' if r.enabled else '⏸️ 禁用',
                    'level_val': r.level,
                    'color': level_info.get('color', '#888888')
                })
            display_df = pd.DataFrame(rule_rows)

            def _style_rule_row(row):
                bg = WARNING_LEVELS.get(row['level_val'], {}).get('bg_color', '#ffffff')
                if row['状态'] == '⏸️ 禁用':
                    bg = '#f5f5f5'
                return [f'background-color:{bg}' for _ in row]

            styled = display_df.style.apply(_style_rule_row, axis=1).hide(['level_val', 'color'], axis=1)
            st.dataframe(styled, use_container_width=True, height=min(400, 80 + len(rule_rows) * 45))

        st.divider()
        col_add, col_edit, col_del, col_toggle = st.columns(4)
        with col_add:
            show_add = st.button("➕ 新增规则", key="show_add_rule")
        with col_edit:
            edit_disabled = len(rules) == 0
            selected_edit_idx = st.selectbox(
                "选择要编辑的规则",
                range(len(rules)) if rules else [0],
                format_func=lambda i: rules[i].name if rules else "-- 无规则 --",
                key="edit_rule_select",
                disabled=edit_disabled
            )
            show_edit = st.button("✏️ 编辑规则", key="show_edit_rule", disabled=edit_disabled)
        with col_del:
            del_disabled = len(rules) == 0
            selected_del_idx = st.selectbox(
                "选择要删除的规则",
                range(len(rules)) if rules else [0],
                format_func=lambda i: rules[i].name if rules else "-- 无规则 --",
                key="del_rule_select",
                disabled=del_disabled
            )
            confirm_del = st.button("🗑️ 删除规则", key="confirm_del_rule", disabled=del_disabled)
            if confirm_del and len(rules) > 0:
                del_name = rules[selected_del_idx].name
                del st.session_state.warning_rules[selected_del_idx]
                st.success(f"✅ 已删除规则: {del_name}")
                st.rerun()
        with col_toggle:
            tog_disabled = len(rules) == 0
            selected_tog_idx = st.selectbox(
                "选择要切换的规则",
                range(len(rules)) if rules else [0],
                format_func=lambda i: rules[i].name if rules else "-- 无规则 --",
                key="toggle_rule_select",
                disabled=tog_disabled
            )
            toggle_btn = st.button("🔄 启用/禁用", key="toggle_rule_btn", disabled=tog_disabled)
            if toggle_btn and len(rules) > 0:
                st.session_state.warning_rules[selected_tog_idx].enabled = not st.session_state.warning_rules[selected_tog_idx].enabled
                st.rerun()

        def _render_rule_form(form_key, default_rule=None, is_edit=False):
            default_name = default_rule.name if default_rule else ""
            default_level = default_rule.level if default_rule else 1
            default_duration = default_rule.duration_minutes if default_rule else 0
            default_enabled = default_rule.enabled if default_rule else True
            default_conditions = default_rule.conditions if default_rule else []
            default_prerequisite = default_rule.prerequisite_rule if default_rule else None

            with st.form(form_key, clear_on_submit=True):
                st.markdown("**规则基本信息**")
                c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                new_name = c1.text_input("规则名称", value=default_name, placeholder="请输入唯一规则名称")
                level_options = [1, 2, 3, 4]
                level_labels = [f"{WARNING_LEVELS[l]['name']} (L{l})" for l in level_options]
                new_level = c2.selectbox(
                    "预警级别",
                    range(len(level_options)),
                    index=level_options.index(default_level) if default_level in level_options else 0,
                    format_func=lambda i: level_labels[i]
                )
                new_duration = c3.number_input("持续时间(分钟)", min_value=0, max_value=1440, value=default_duration,
                                                help="0表示无需持续，单条满足即触发")
                new_enabled = c4.checkbox("启用规则", value=default_enabled)

                st.markdown("**前置依赖规则**")
                existing_rule_names = [r.name for j, r in enumerate(st.session_state.warning_rules)
                                       if not is_edit or j != selected_edit_idx]
                prereq_options = ["无"] + existing_rule_names
                default_prereq_idx = 0
                if default_prerequisite and default_prerequisite in prereq_options:
                    default_prereq_idx = prereq_options.index(default_prerequisite)
                new_prereq = st.selectbox(
                    "前置规则（单选，可不选）",
                    range(len(prereq_options)),
                    index=default_prereq_idx,
                    format_func=lambda i: prereq_options[i],
                    help="设置了前置规则的规则只有在前置规则已触发时才会被检测"
                )

                st.markdown("**触发条件（所有条件需同时满足）**")
                cond_count_key = f"{form_key}_cond_count"
                if cond_count_key not in st.session_state:
                    st.session_state[cond_count_key] = max(1, len(default_conditions))

                def _change_cond_count(delta):
                    st.session_state[cond_count_key] = max(1, min(10, st.session_state[cond_count_key] + delta))

                cond_cols = st.columns([1, 1, 1, 2, 1, 1])
                cond_cols[0].markdown("**序号**")
                cond_cols[1].markdown("**参数**")
                cond_cols[2].markdown("**比较符**")
                cond_cols[3].markdown("**阈值**")
                cond_cols[4].markdown("**操作**")
                cond_cols[5].markdown("")

                new_conditions = []
                valid_form = True
                for i in range(st.session_state[cond_count_key]):
                    cc = st.columns([1, 1, 1, 2, 1, 1])
                    cc[0].markdown(f"**{i+1}**")
                    default_cond = default_conditions[i] if i < len(default_conditions) else None
                    dp = default_cond.param if default_cond else AVAILABLE_PARAMS[0]
                    if dp not in AVAILABLE_PARAMS:
                        dp = AVAILABLE_PARAMS[0]
                    p_idx = AVAILABLE_PARAMS.index(dp) if dp in AVAILABLE_PARAMS else 0
                    cond_param = cc[1].selectbox(
                        "参数", AVAILABLE_PARAMS, index=p_idx,
                        format_func=lambda x: PARAM_NAMES_CN.get(x, x),
                        key=f"{form_key}_p{i}", label_visibility="collapsed"
                    )
                    do = default_cond.operator if default_cond else ">"
                    if do not in OPERATORS:
                        do = ">"
                    o_idx = OPERATORS.index(do) if do in OPERATORS else 0
                    cond_op = cc[2].selectbox(
                        "比较符", OPERATORS, index=o_idx,
                        key=f"{form_key}_o{i}", label_visibility="collapsed"
                    )
                    dt = default_cond.threshold if default_cond else 0.0
                    try:
                        dt_val = float(dt)
                    except (ValueError, TypeError):
                        dt_val = 0.0
                    cond_thresh = cc[3].number_input(
                        "阈值", value=dt_val, format="%.4f",
                        key=f"{form_key}_t{i}", label_visibility="collapsed"
                    )
                    new_conditions.append(Condition(param=cond_param, operator=cond_op, threshold=float(cond_thresh)))
                    if i == st.session_state[cond_count_key] - 1:
                        if cc[4].button("➕", key=f"{form_key}_add{i}", help="添加条件"):
                            _change_cond_count(1)
                            st.rerun()
                    else:
                        if cc[4].button("➖", key=f"{form_key}_del{i}", help="删除此条件"):
                            _change_cond_count(-1)
                            st.rerun()

                st.divider()
                submit_label = "💾 保存修改" if is_edit else "✅ 确认添加"
                submitted = st.form_submit_button(submit_label, type="primary")
                if submitted:
                    if not new_name.strip():
                        st.error("❌ 规则名称不能为空")
                        valid_form = False
                    else:
                        existing_names = [r.name for j, r in enumerate(st.session_state.warning_rules)
                                          if not is_edit or j != selected_edit_idx]
                        if new_name.strip() in existing_names:
                            st.error("❌ 规则名称已存在，请使用其他名称")
                            valid_form = False

                    selected_prereq = prereq_options[new_prereq] if new_prereq > 0 else None
                    if valid_form and selected_prereq is not None:
                        temp_rules = [r for j, r in enumerate(st.session_state.warning_rules)
                                      if not is_edit or j != selected_edit_idx]
                        if detect_cycle_dependency(temp_rules, new_name.strip(), selected_prereq):
                            st.error("❌ 检测到循环依赖！设置此外前置规则会导致循环依赖，请重新选择")
                            valid_form = False

                    if valid_form:
                        level_val = level_options[new_level]
                        selected_prereq_val = prereq_options[new_prereq] if new_prereq > 0 else None
                        new_rule = WarningRule(
                            name=new_name.strip(),
                            level=level_val,
                            conditions=new_conditions,
                            duration_minutes=int(new_duration),
                            enabled=new_enabled,
                            prerequisite_rule=selected_prereq_val
                        )
                        if is_edit:
                            st.session_state.warning_rules[selected_edit_idx] = new_rule
                            st.success(f"✅ 规则已更新: {new_name}")
                        else:
                            st.session_state.warning_rules.append(new_rule)
                            st.success(f"✅ 规则已添加: {new_name}")
                        if cond_count_key in st.session_state:
                            del st.session_state[cond_count_key]
                        st.rerun()

        if show_add:
            st.markdown("---")
            st.subheader("➕ 新增预警规则")
            _render_rule_form("add_rule_form", default_rule=None, is_edit=False)

        if show_edit and len(rules) > 0:
            st.markdown("---")
            st.subheader(f"✏️ 编辑规则: {rules[selected_edit_idx].name}")
            _render_rule_form("edit_rule_form", default_rule=rules[selected_edit_idx], is_edit=True)

    with tab_we:
        st.subheader("实时预警扫描")
        col_scan1, col_scan2, col_scan3 = st.columns([1, 1, 3])
        with col_scan1:
            scan_btn = st.button("🚀 执行预警扫描", type="primary")
        with col_scan2:
            clear_btn = st.button("🗑️ 清空扫描结果")
            if clear_btn:
                st.session_state.warning_events = None
                st.session_state.warning_events_df = None
                st.session_state.warning_composite_groups = []
                st.session_state.warning_escalation_state = {}
                st.rerun()
        with col_scan3:
            enabled_count = len([r for r in st.session_state.warning_rules if r.enabled])
            st.info(f"当前已启用规则: {enabled_count}/{len(st.session_state.warning_rules)} 条")

        if scan_btn:
            if st.session_state.data is None:
                st.error("❌ 请先在【数据导入】模块上传数据")
            else:
                data_to_scan = st.session_state.data.copy()
                if st.session_state.qc_result is not None:
                    qc_codes = st.session_state.qc_result.qc_codes
                    if qc_codes is not None:
                        pass
                with st.spinner("正在执行预警扫描..."):
                    events = scan_warnings(
                        data_to_scan,
                        st.session_state.warning_rules,
                        st.session_state.qc_result.qc_codes if st.session_state.qc_result else None
                    )
                    all_buoys = data_to_scan['buoy_id'].unique().tolist()
                    events, new_state = apply_escalation(
                        events,
                        st.session_state.warning_rules,
                        all_buoys,
                        st.session_state.warning_escalation_state
                    )
                    st.session_state.warning_escalation_state = new_state
                    events, composite_groups = build_composite_groups(events)
                    st.session_state.warning_events = events
                    st.session_state.warning_events_df = events_to_dataframe(events)
                    st.session_state.warning_composite_groups = composite_groups
                    group_count = len(composite_groups)
                    st.success(f"✅ 扫描完成！共检测到 {len(events)} 个预警事件，{group_count} 个复合事件组")

        events_df = st.session_state.warning_events_df
        if events_df is None or len(events_df) == 0:
            if scan_btn:
                st.info("未检测到任何预警事件")
            else:
                st.info("请点击【执行预警扫描】开始检测")
        else:
            st.success(f"共检测到 **{len(events_df)}** 个预警事件")
            st.divider()
            st.subheader("事件筛选")
            col_f1, col_f2, col_f3 = st.columns(3)
            eff_levels = sorted(events_df['effective_level'].unique().tolist()) if 'effective_level' in events_df.columns else sorted(events_df['level'].unique().tolist())
            all_levels = ['全部'] + eff_levels
            level_labels = ['全部'] + [f"L{l} - {WARNING_LEVELS.get(l, {}).get('name', '')}" for l in all_levels if l != '全部']
            with col_f1:
                sel_level_idx = st.selectbox(
                    "按有效级别筛选",
                    range(len(all_levels)),
                    format_func=lambda i: level_labels[i],
                    key="event_filter_level"
                )
            all_buoys = ['全部'] + sorted(events_df['buoy_id'].unique().tolist())
            with col_f2:
                sel_buoy = st.selectbox(
                    "按浮标筛选",
                    all_buoys,
                    key="event_filter_buoy"
                )
            all_tags = ['全部', '↑升级', '↓原始']
            with col_f3:
                sel_tag = st.selectbox(
                    "按升降级标签筛选",
                    all_tags,
                    key="event_filter_tag"
                )

            filtered = events_df.copy()
            level_col = 'effective_level' if 'effective_level' in filtered.columns else 'level'
            if all_levels[sel_level_idx] != '全部':
                filtered = filtered[filtered[level_col] == all_levels[sel_level_idx]]
            if sel_buoy != '全部':
                filtered = filtered[filtered['buoy_id'] == sel_buoy]
            if sel_tag != '全部' and 'level_tag' in filtered.columns:
                filtered = filtered[filtered['level_tag'] == sel_tag]

            if len(filtered) == 0:
                st.warning("筛选条件下无事件")
            else:
                st.markdown(f"显示 **{len(filtered)}** 条事件")

                def _row_style(row):
                    eff_lv = row.get('effective_level', row['level'])
                    bg = WARNING_LEVELS.get(eff_lv, {}).get('bg_color', '#ffffff')
                    if eff_lv >= 3:
                        return [f'background-color:{bg};font-weight:bold' for _ in row]
                    return [f'background-color:{bg}' for _ in row]

                display = filtered.copy()
                display['触发时间'] = display['start_time'].dt.strftime('%Y-%m-%d %H:%M')
                display['结束时间'] = display['end_time'].dt.strftime('%Y-%m-%d %H:%M')
                display['持续(分钟)'] = display['duration_minutes']
                eff_col = 'effective_level' if 'effective_level' in display.columns else 'level'
                eff_name_col = 'effective_level_name' if 'effective_level_name' in display.columns else 'level_name'
                display['有效级别'] = display.apply(
                    lambda r: f"{r.get(eff_name_col, WARNING_LEVELS.get(r.get(eff_col, r['level']), {}).get('name', ''))} {r.get('level_tag', '')}", axis=1
                )
                display['参数快照'] = display['param_snapshot'].apply(
                    lambda d: " | ".join([f"{PARAM_NAMES_CN.get(k, k)}: {v:.2f}" if pd.notna(v) else f"{PARAM_NAMES_CN.get(k, k)}: N/A"
                                          for k, v in d.items()])
                )
                display['复合组'] = display['group_id'].apply(lambda x: x if pd.notna(x) else "-")
                show_cols = ['event_id', 'rule_name', '有效级别', 'buoy_id', '触发时间', '结束时间', '持续(分钟)', '复合组', '参数快照']
                show_cols_rename = {
                    'event_id': '事件ID',
                    'rule_name': '触发规则',
                    'buoy_id': '浮标ID'
                }
                display_final = display[show_cols].rename(columns=show_cols_rename)
                styled_final = display_final.style.apply(_row_style, axis=1)
                st.dataframe(styled_final, use_container_width=True, height=500)

                composite_groups = st.session_state.warning_composite_groups
                if composite_groups:
                    st.divider()
                    st.subheader("🔗 复合事件组")
                    for grp in composite_groups:
                        max_lv = grp['max_level']
                        max_lv_info = WARNING_LEVELS.get(max_lv, {})
                        max_lv_color = max_lv_info.get('color', '#888')
                        max_lv_name = max_lv_info.get('name', '')
                        time_range = f"{grp['start_time'].strftime('%Y-%m-%d %H:%M')} ~ {grp['end_time'].strftime('%Y-%m-%d %H:%M')}"
                        title = f"{grp['group_id']} | {max_lv_name}预警 | {grp['rule_count']}条规则 | {time_range}"
                        with st.expander(f"🔗 {title}"):
                            member_dicts = grp.get('events_dicts', [])
                            for sub in member_dicts:
                                sub_lv = sub.get('effective_level', sub.get('level', 1))
                                sub_lv_info = WARNING_LEVELS.get(int(sub_lv), {})
                                sub_color = sub_lv_info.get('color', '#888')
                                sub_bg = sub_lv_info.get('bg_color', '#fff')
                                tag_str = sub.get('level_tag', '')
                                param_snap = sub.get('param_snapshot', {}) or {}
                                param_html = ""
                                for pk, pv in param_snap.items():
                                    cn_name = PARAM_NAMES_CN.get(pk, pk)
                                    try:
                                        val_str = f"{float(pv):.3f}" if pv is not None and pd.notna(pv) else "N/A"
                                    except (ValueError, TypeError):
                                        val_str = str(pv) if pv is not None else "N/A"
                                    param_html += f'<span style="margin-right:12px;">{cn_name}: <b>{val_str}</b></span>'
                                dur_min = sub.get('duration_minutes', 0)
                                dur_str = f"{dur_min}分钟" if dur_min > 0 else "瞬时触发"
                                start_t = sub.get('start_time')
                                start_str = start_t.strftime('%Y-%m-%d %H:%M') if start_t is not None and hasattr(start_t, 'strftime') else str(start_t)
                                rule_name = sub.get('rule_name', '')
                                sub_card = f"""
                                <div style="border:1px solid {sub_color};border-radius:8px;padding:12px;margin-bottom:8px;background:{sub_bg};">
                                    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                                        <span style="font-weight:bold;">{rule_name}</span>
                                        <span style="background:{sub_color};color:white;padding:2px 8px;border-radius:4px;font-size:12px;">{sub_lv_info.get('name', '')}</span>
                                        <span style="background:#e0e0e0;color:#333;padding:2px 8px;border-radius:4px;font-size:12px;">{tag_str}</span>
                                        <span style="font-size:13px;">⏰ {start_str}</span>
                                        <span style="font-size:13px;">📏 {dur_str}</span>
                                    </div>
                                    <div style="font-size:12px;margin-top:6px;">📊 {param_html if param_html else '无参数快照'}</div>
                                </div>
                                """
                                st.markdown(sub_card, unsafe_allow_html=True)

                st.divider()
                st.subheader("📜 事件卡片详情")
                page_size = 10
                total_pages = max(1, (len(filtered) - 1) // page_size + 1)
                cp_col1, cp_col2, cp_col3 = st.columns([1, 3, 1])
                current_page = cp_col2.slider("页码", 1, total_pages, 1, key="event_page")
                start = (current_page - 1) * page_size
                end = min(start + page_size, len(filtered))
                page_data = filtered.iloc[start:end].reset_index(drop=True)
                st.markdown(f"第 **{current_page}/{total_pages}** 页，显示 **{start+1}-{end}** 条")

                for _, ev in page_data.iterrows():
                    eff_lv = int(ev.get('effective_level', ev['level']))
                    orig_lv = int(ev['level'])
                    eff_level_info = WARNING_LEVELS.get(eff_lv, {})
                    orig_level_info = WARNING_LEVELS.get(orig_lv, {})
                    color = eff_level_info.get('color', '#888')
                    bg = eff_level_info.get('bg_color', '#fff')
                    border_style = f'3px solid {color}' if eff_lv >= 3 else f'1px solid {color}'
                    param_html = ""
                    for pk, pv in ev['param_snapshot'].items():
                        cn_name = PARAM_NAMES_CN.get(pk, pk)
                        val_str = f"{pv:.3f}" if pd.notna(pv) else "N/A"
                        param_html += f'<span style="margin-right:18px;">{cn_name}: <b>{val_str}</b></span>'

                    dur_str = f"{ev['duration_minutes']}分钟" if ev['duration_minutes'] > 0 else "瞬时触发"
                    tag_html = ""
                    level_tag = ev.get('level_tag', '')
                    if level_tag == '↑升级':
                        tag_html = f'<span style="background:#FF4444;color:white;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:bold;">↑升级</span>'
                        level_detail = f"{orig_level_info.get('name', '')}→{eff_level_info.get('name', '')}"
                    elif level_tag == '↓原始':
                        tag_html = f'<span style="background:#999;color:white;padding:2px 8px;border-radius:4px;font-size:11px;">↓原始</span>'
                        level_detail = ""
                    else:
                        level_detail = ""

                    group_html = ""
                    group_id = ev.get('group_id', None)
                    if pd.notna(group_id) and group_id:
                        group_html = f'<span style="background:#6A5ACD;color:white;padding:2px 8px;border-radius:5px;font-size:12px;">🔗 {group_id}</span>'

                    card = f"""
                    <div style="border:{border_style};border-radius:10px;padding:16px;margin-bottom:12px;background:{bg};box-shadow:0 2px 4px rgba(0,0,0,0.05);">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:8px;">
                            <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                                <span style="background:#333;color:white;padding:3px 10px;border-radius:5px;font-size:13px;font-weight:bold;">#{ev['event_id']}</span>
                                <span style="font-weight:bold;font-size:16px;">{ev['rule_name']}</span>
                                <span style="background:{color};color:white;padding:3px 12px;border-radius:5px;font-size:13px;font-weight:bold;">{eff_level_info.get('name', '')}预警</span>
                                {tag_html}
                                <span style="font-size:12px;color:#666;">{level_detail}</span>
                                <span style="background:#666;color:white;padding:3px 10px;border-radius:5px;font-size:12px;">🚩 {ev['buoy_id']}</span>
                                {group_html}
                            </div>
                        </div>
                        <div style="display:flex;gap:24px;font-size:14px;margin-bottom:10px;flex-wrap:wrap;">
                            <span>⏰ 开始: <b>{ev['start_time'].strftime('%Y-%m-%d %H:%M:%S')}</b></span>
                            <span>⏱️ 结束: <b>{ev['end_time'].strftime('%Y-%m-%d %H:%M:%S')}</b></span>
                            <span>📏 持续: <b>{dur_str}</b></span>
                        </div>
                        <div style="font-size:13px;padding-top:8px;border-top:1px dashed #ccc;">
                            📊 触发时参数: {param_html}
                        </div>
                    </div>
                    """
                    st.markdown(card, unsafe_allow_html=True)

    with tab_ws:
        st.subheader("预警统计仪表盘")
        stats_df = st.session_state.warning_events_df
        if stats_df is None or len(stats_df) == 0:
            st.info("暂无统计数据，请先在【预警事件列表】中执行扫描")
        else:
            eff_col = 'effective_level' if 'effective_level' in stats_df.columns else 'level'
            sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
            total = len(stats_df)
            red_count = len(stats_df[stats_df[eff_col] == 4])
            orange_count = len(stats_df[stats_df[eff_col] == 3])
            yellow_count = len(stats_df[stats_df[eff_col] == 2])
            blue_count = len(stats_df[stats_df[eff_col] == 1])
            composite_groups = st.session_state.warning_composite_groups
            group_count = len(composite_groups) if composite_groups else 0
            sc1.metric("🚨 总预警数", total)
            sc2.metric("🔴 红色", red_count)
            sc3.metric("🟠 橙色", orange_count)
            sc4.metric("🟡 黄色", yellow_count)
            sc5.metric("🔵 蓝色", blue_count)
            sc6.metric("🔗 复合事件组", group_count)

            st.divider()
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("**各级别预警数量分布（按有效级别）**")
                level_counts = compute_level_counts(stats_df, use_effective=True)
                if len(level_counts) > 0:
                    level_counts['label'] = level_counts.apply(
                        lambda r: f"L{r['level']} {r['level_name']}", axis=1
                    )
                    level_counts['color'] = level_counts['level'].map(
                        lambda l: WARNING_LEVELS.get(int(l), {}).get('color', '#999')
                    )
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=level_counts['label'],
                        values=level_counts['count'],
                        marker_colors=level_counts['color'].tolist(),
                        textinfo='label+percent+value',
                        hole=0.4,
                        sort=False
                    )])
                    fig_pie.update_layout(height=420, showlegend=True,
                                          legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5))
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("无数据")

            with chart_col2:
                st.markdown("**各浮标预警频次**")
                buoy_counts = compute_buoy_counts(stats_df)
                if len(buoy_counts) > 0:
                    bar_colors = []
                    for _, row in buoy_counts.iterrows():
                        bid = row['buoy_id']
                        buoy_df = stats_df[stats_df['buoy_id'] == bid]
                        max_lv = buoy_df[eff_col].max()
                        bar_colors.append(WARNING_LEVELS.get(int(max_lv), {}).get('color', '#4A90D9'))

                    fig_bar = go.Figure(data=[go.Bar(
                        x=buoy_counts['buoy_id'],
                        y=buoy_counts['count'],
                        marker_color=bar_colors,
                        text=buoy_counts['count'],
                        textposition='outside',
                        hovertemplate='浮标: %{x}<br>预警次数: %{y}<extra></extra>'
                    )])
                    fig_bar.update_layout(height=420,
                                          xaxis_title="浮标ID",
                                          yaxis_title="预警次数",
                                          yaxis=dict(rangemode='tozero'))
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("无数据")

            st.divider()
            st.markdown("**24小时内预警时间分布热力图**")
            all_buoys = sorted(stats_df['buoy_id'].unique().tolist()) if 'buoy_id' in stats_df.columns else []
            heatmap_df = compute_hourly_heatmap(stats_df, all_buoys)
            if len(heatmap_df) > 0:
                fig_heat = go.Figure(data=go.Heatmap(
                    z=heatmap_df.values,
                    x=[f"{h:02d}:00" for h in range(24)],
                    y=heatmap_df.index.tolist(),
                    colorscale='Reds',
                    showscale=True,
                    hoverongaps=False,
                    text=[[str(int(v)) if pd.notna(v) else "0" for v in row] for row in heatmap_df.values],
                    texttemplate="%{text}",
                    hovertemplate='浮标: %{y}<br>时段: %{x}<br>预警次数: %{z}<extra></extra>'
                ))
                max_val = int(heatmap_df.values.max()) if len(heatmap_df) > 0 else 0
                if max_val <= 5:
                    tick_vals = list(range(max_val + 1))
                else:
                    tick_vals = list(range(0, max_val + 1, max(1, max_val // 5)))
                fig_heat.update_layout(
                    height=max(300, 80 + len(heatmap_df) * 45),
                    xaxis_title="时段",
                    yaxis_title="浮标ID",
                    xaxis=dict(dtick=2),
                    coloraxis_colorbar=dict(
                        title="预警次数",
                        tickvals=tick_vals,
                        ticktext=[str(v) for v in tick_vals]
                    )
                )
                st.plotly_chart(fig_heat, use_container_width=True)

                with st.expander("📋 查看热力图数据明细"):
                    display_heat = heatmap_df.copy()
                    display_heat.columns = [f"{h:02d}时" for h in range(24)]
                    display_heat.index.name = "浮标ID"
                    st.dataframe(display_heat.style.background_gradient(cmap='Reds', axis=None), use_container_width=True)
            else:
                st.info("无数据")

            st.divider()
            st.subheader("📈 历史趋势")
            daily_trend = compute_daily_trend(stats_df)
            daily_totals = compute_daily_totals(stats_df)
            ma_df = compute_7day_moving_avg(daily_totals)
            anomaly_days = detect_anomaly_days(ma_df)

            if len(daily_trend) > 0 or len(daily_totals) > 0:
                fig_trend = go.Figure()

                if len(daily_trend) > 0:
                    sorted_levels = sorted(daily_trend['level'].unique())
                    for lv in sorted_levels:
                        lv_data = daily_trend[daily_trend['level'] == lv].sort_values('date')
                        lv_info = WARNING_LEVELS.get(int(lv), {})
                        fig_trend.add_trace(go.Scatter(
                            x=lv_data['date'],
                            y=lv_data['count'],
                            name=f"L{int(lv)} {lv_info.get('name', '')}",
                            mode='lines+markers',
                            stackgroup='one',
                            line=dict(color=lv_info.get('color', '#888')),
                            hovertemplate=f'L{int(lv)} {lv_info.get("name", "")}<br>日期: %{{x}}<br>事件数: %{{y}}<extra></extra>'
                        ))

                if len(ma_df) > 0 and 'ma7' in ma_df.columns:
                    fig_trend.add_trace(go.Scatter(
                        x=ma_df['date'],
                        y=ma_df['ma7'],
                        name='7日滑动平均',
                        mode='lines',
                        line=dict(color='#333333', width=2, dash='dash'),
                        hovertemplate='7日滑动平均<br>日期: %{x}<br>均值: %{y:.1f}<extra></extra>'
                    ))

                if anomaly_days:
                    anomaly_df = ma_df[ma_df['date'].isin(anomaly_days)]
                    fig_trend.add_trace(go.Scatter(
                        x=anomaly_df['date'],
                        y=anomaly_df['total_count'],
                        name='异常',
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=14,
                            color='red',
                            line=dict(width=2, color='darkred')
                        ),
                        hovertemplate='⚠️ 异常<br>日期: %{x}<br>事件数: %{y}<extra></extra>'
                    ))

                fig_trend.update_layout(
                    height=500,
                    xaxis_title="日期",
                    yaxis_title="事件数",
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_trend, use_container_width=True)

                if anomaly_days:
                    st.warning(f"⚠️ 检测到 {len(anomaly_days)} 个异常日（事件数超过7日均值的2倍）")
                    for ad in anomaly_days:
                        row_data = ma_df[ma_df['date'] == ad]
                        if len(row_data) > 0:
                            cnt = int(row_data.iloc[0]['total_count'])
                            avg = row_data.iloc[0]['ma7']
                            st.markdown(f"  🔺 {ad}: {cnt}次事件 (7日均值: {avg:.1f})")

                with st.expander("📋 查看趋势数据明细"):
                    if len(ma_df) > 0:
                        display_trend = ma_df.copy()
                        display_trend['异常'] = display_trend['date'].apply(
                            lambda d: '⚠️ 是' if d in anomaly_days else ''
                        )
                        display_trend['ma7'] = display_trend['ma7'].round(1)
                        display_trend.columns = ['日期', '事件数', '7日滑动平均', '异常']
                        st.dataframe(display_trend.style.apply(
                            lambda row: ['background-color:#FFE6E6' if row['异常'] else '' for _ in row],
                            axis=1
                        ), use_container_width=True)
            else:
                st.info("数据不足，无法生成趋势图")

elif page == "📄 报告生成":
    st.header("PDF海况分析报告")
    if st.session_state.data is None:
        st.warning("请先在【数据导入】模块上传数据")
    else:
        buoy_ids = st.session_state.data['buoy_id'].unique()
        col1, col2, col3 = st.columns(3)
        selected_buoy = col1.selectbox("选择浮标", buoy_ids)
        report_id = col2.text_input("报告编号", value="RPT-" + pd.Timestamp.now().strftime("%Y%m%d-%H%M"))
        output_filename = col3.text_input("输出文件名", value=f"海况分析报告_{selected_buoy}.pdf")
        overview = st.session_state.data_overview
        if selected_buoy in overview:
            info = overview[selected_buoy]
            st.info(f"时间范围: {info['time_start']} ~ {info['time_end']} | 记录数: {info['record_count']}")
        if st.button("📄 生成PDF报告", type="primary"):
            with st.spinner("正在生成PDF报告..."):
                wave_figs = []
                current_figs = []
                extreme_figs = []
                met_figs = []
                try:
                    wave_result = analyze_wave_buoy(
                        st.session_state.data, selected_buoy,
                        st.session_state.qc_result.qc_codes if st.session_state.qc_result else None
                    )
                    if 'spectrum_fig' in wave_result:
                        wave_figs.append(wave_result['spectrum_fig'])
                    if 'dir_spectrum_fig' in wave_result:
                        wave_figs.append(wave_result['dir_spectrum_fig'])
                except Exception:
                    pass
                try:
                    current_result = analyze_current_buoy(
                        st.session_state.data, selected_buoy,
                        st.session_state.qc_result.qc_codes if st.session_state.qc_result else None
                    )
                    if 'tidal_ellipses_fig' in current_result:
                        current_figs.append(current_result['tidal_ellipses_fig'])
                    if 'residual_fig' in current_result:
                        current_figs.append(current_result['residual_fig'])
                    if 'current_rose_fig' in current_result:
                        current_figs.append(current_result['current_rose_fig'])
                except Exception:
                    pass
                try:
                    extreme_result = analyze_extremes(
                        st.session_state.data, selected_buoy,
                        st.session_state.qc_result.qc_codes if st.session_state.qc_result else None
                    )
                    if 'return_curve_fig' in extreme_result:
                        extreme_figs.append(extreme_result['return_curve_fig'])
                    if 'histogram_fig' in extreme_result:
                        extreme_figs.append(extreme_result['histogram_fig'])
                except Exception:
                    pass
                try:
                    met_result = analyze_meteorology_buoy(
                        st.session_state.data, selected_buoy,
                        st.session_state.qc_result.qc_codes if st.session_state.qc_result else None
                    )
                    if 'wind_rose' in met_result:
                        met_figs.append(met_result['wind_rose'])
                except Exception:
                    pass
                try:
                    generate_pdf_report(
                        output_filename, selected_buoy,
                        str(info['time_start']), str(info['time_end']),
                        report_id, overview,
                        st.session_state.qc_result.level_stats if st.session_state.qc_result else {},
                        wave_figs, current_figs, extreme_figs, met_figs
                    )
                    with open(output_filename, 'rb') as f:
                        pdf_data = f.read()
                    st.success(f"✅ 报告生成成功: {output_filename}")
                    st.download_button(
                        label="📥 下载PDF报告",
                        data=pdf_data,
                        file_name=output_filename,
                        mime='application/pdf'
                    )
                except Exception as e:
                    st.error(f"报告生成失败: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
