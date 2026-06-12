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

st.set_page_config(page_title="海洋浮标数据质控与海况分析系统", layout="wide")
st.title("🌊 海洋浮标观测数据质量控制与海况分析系统")


if 'data' not in st.session_state:
    st.session_state.data = None
if 'qc_result' not in st.session_state:
    st.session_state.qc_result = None
if 'data_overview' not in st.session_state:
    st.session_state.data_overview = None


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
