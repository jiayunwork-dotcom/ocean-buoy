import numpy as np
import pandas as pd
from scipy import stats, optimize
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .plot_utils import setup_chinese_font
setup_chinese_font()

RETURN_PERIODS = [10, 25, 50, 100]


def extract_annual_maxima(hs_series: pd.Series) -> pd.Series:
    if hs_series.index.name != 'time':
        hs_series = hs_series.copy()
        hs_series.index = hs_series['time'] if 'time' in hs_series else hs_series.index
    annual_max = hs_series.dropna().resample('YE').max()
    annual_max = annual_max.dropna()
    return annual_max


def fit_gumbel(data: np.ndarray) -> Dict:
    params = stats.gumbel_r.fit(data)
    loc, scale = params
    n = len(data)
    log_likelihood = np.sum(stats.gumbel_r.logpdf(data, loc=loc, scale=scale))
    aic = 2 * 2 - 2 * log_likelihood
    return {
        'distribution': 'Gumbel',
        'params': {'loc': loc, 'scale': scale},
        'log_likelihood': log_likelihood,
        'aic': aic
    }


def fit_weibull_3p(data: np.ndarray) -> Dict:
    try:
        params = stats.weibull_min.fit(data, floc=0)
        if len(params) == 3:
            c, loc, scale = params
        else:
            c, scale = params
            loc = 0
        n = len(data)
        log_likelihood = np.sum(stats.weibull_min.logpdf(data, c=c, loc=loc, scale=scale))
        aic = 2 * 3 - 2 * log_likelihood
        return {
            'distribution': 'Weibull3P',
            'params': {'c': c, 'loc': loc, 'scale': scale},
            'log_likelihood': log_likelihood,
            'aic': aic
        }
    except Exception:
        return fit_weibull_2p(data)


def fit_weibull_2p(data: np.ndarray) -> Dict:
    params = stats.weibull_min.fit(data, floc=0)
    c, _, scale = params
    n = len(data)
    log_likelihood = np.sum(stats.weibull_min.logpdf(data, c=c, loc=0, scale=scale))
    aic = 2 * 2 - 2 * log_likelihood
    return {
        'distribution': 'Weibull2P',
        'params': {'c': c, 'loc': 0, 'scale': scale},
        'log_likelihood': log_likelihood,
        'aic': aic
    }


def ks_test(data: np.ndarray, fit_result: Dict) -> Dict:
    dist = fit_result['distribution']
    params = fit_result['params']
    if dist == 'Gumbel':
        cdf_func = lambda x: stats.gumbel_r.cdf(x, loc=params['loc'], scale=params['scale'])
    elif dist.startswith('Weibull'):
        cdf_func = lambda x: stats.weibull_min.cdf(x, c=params['c'], loc=params.get('loc', 0), scale=params['scale'])
    else:
        return {'D': np.nan, 'p_value': np.nan}
    D, p_value = stats.kstest(data, cdf_func)
    return {'D': D, 'p_value': p_value}


def compute_return_level(fit_result: Dict, return_period: float) -> float:
    p = 1.0 - 1.0 / return_period
    dist = fit_result['distribution']
    params = fit_result['params']
    if dist == 'Gumbel':
        return stats.gumbel_r.ppf(p, loc=params['loc'], scale=params['scale'])
    elif dist.startswith('Weibull'):
        return stats.weibull_min.ppf(p, c=params['c'], loc=params.get('loc', 0), scale=params['scale'])
    return np.nan


def profile_likelihood_ci(data: np.ndarray, fit_result: Dict, return_period: float,
                          alpha: float = 0.05) -> Tuple[float, float]:
    target_level = compute_return_level(fit_result, return_period)
    max_ll = fit_result['log_likelihood']
    critical = stats.chi2.ppf(1 - alpha, df=1) / 2
    dist = fit_result['distribution']
    params = fit_result['params']
    def neg_profile_ll(level):
        if dist == 'Gumbel':
            def objective(params_arr):
                loc, scale = params_arr
                if scale <= 0:
                    return 1e10
                p_val = 1.0 - 1.0 / return_period
                level_constraint = stats.gumbel_r.ppf(p_val, loc=loc, scale=scale) - level
                if abs(level_constraint) > 1e-6:
                    return 1e10
                return -np.sum(stats.gumbel_r.logpdf(data, loc=loc, scale=scale))
            try:
                res = optimize.minimize(objective, [params['loc'], params['scale']],
                                        method='Nelder-Mead', options={'maxiter': 1000})
                return -res.fun if res.success else -max_ll
            except Exception:
                return -max_ll
        elif dist.startswith('Weibull'):
            def objective(params_arr):
                c, scale = params_arr
                if c <= 0 or scale <= 0:
                    return 1e10
                p_val = 1.0 - 1.0 / return_period
                level_constraint = stats.weibull_min.ppf(p_val, c=c, loc=params.get('loc', 0), scale=scale) - level
                if abs(level_constraint) > 1e-6:
                    return 1e10
                return -np.sum(stats.weibull_min.logpdf(data, c=c, loc=params.get('loc', 0), scale=scale))
            try:
                res = optimize.minimize(objective, [params['c'], params['scale']],
                                        method='Nelder-Mead', options={'maxiter': 1000})
                return -res.fun if res.success else -max_ll
            except Exception:
                return -max_ll
        return -max_ll
    try:
        std_error = np.std(data) / np.sqrt(len(data))
        lower_guess = max(0.01, target_level - 3 * std_error)
        upper_guess = target_level + 5 * std_error
        def find_ci_bound(start_level, direction):
            level = start_level
            step = std_error * direction * 0.5
            for _ in range(50):
                ll = profile_ll_wrapper(level, data, fit_result, return_period)
                if max_ll - ll > critical:
                    break
                level += step
            return level
        def profile_ll_wrapper(level, data, fit_result, return_period):
            return neg_profile_ll(level)
        lower = find_ci_bound(target_level, -1)
        upper = find_ci_bound(target_level, 1)
        return (max(0, lower), upper)
    except Exception:
        std_error = np.std(data) / np.sqrt(len(data))
        return (max(0, target_level - 1.96 * std_error), target_level + 1.96 * std_error)


def plot_return_curve(data: np.ndarray, fit_results: List[Dict], return_periods: List[int] = None) -> Figure:
    if return_periods is None:
        return_periods = RETURN_PERIODS
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_data = np.sort(data)
    n = len(data)
    empirical_rp = 1.0 / (1.0 - np.arange(1, n + 1) / (n + 1))
    ax.scatter(empirical_rp, sorted_data, color='k', label='经验点', alpha=0.7, s=30)
    rp_range = np.logspace(np.log10(1), np.log10(max(return_periods) * 2), 100)
    colors = ['r', 'b']
    for i, fit_result in enumerate(fit_results):
        levels = [compute_return_level(fit_result, rp) for rp in rp_range]
        ax.plot(rp_range, levels, colors[i], linewidth=2, label=fit_result['distribution'])
        for rp in return_periods:
            level = compute_return_level(fit_result, rp)
            ax.scatter([rp], [level], color=colors[i], marker='s', s=60, zorder=5)
            ax.annotate(f'{rp}年: {level:.2f}m', (rp, level), textcoords="offset points",
                        xytext=(10, 0), fontsize=9, color=colors[i])
    ax.set_xscale('log')
    ax.set_xlabel('重现期 (年)')
    ax.set_ylabel('波高 Hs (m)')
    ax.set_title('重现期-波高关系曲线')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    return fig


def plot_qq(data: np.ndarray, fit_result: Dict) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 7))
    sorted_data = np.sort(data)
    n = len(sorted_data)
    probs = np.arange(1, n + 1) / (n + 1)
    dist = fit_result['distribution']
    params = fit_result['params']
    if dist == 'Gumbel':
        theoretical = stats.gumbel_r.ppf(probs, loc=params['loc'], scale=params['scale'])
    elif dist.startswith('Weibull'):
        theoretical = stats.weibull_min.ppf(probs, c=params['c'], loc=params.get('loc', 0), scale=params['scale'])
    else:
        theoretical = probs
    min_val = min(sorted_data.min(), theoretical.min())
    max_val = max(sorted_data.max(), theoretical.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1线')
    ax.scatter(theoretical, sorted_data, alpha=0.7, s=30)
    ax.set_xlabel('理论分位数')
    ax.set_ylabel('样本分位数')
    ax.set_title(f'Q-Q图 ({fit_result["distribution"]})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig


def plot_histogram(data: np.ndarray) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins='auto', edgecolor='black', alpha=0.7, density=True)
    ax.set_xlabel('年最大有效波高 (m)')
    ax.set_ylabel('频率密度')
    ax.set_title('年最大有效波高频率直方图')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def analyze_extremes(df: pd.DataFrame, buoy_id: str, qc_mask: Optional[pd.DataFrame] = None) -> Dict:
    buoy_df = df[df['buoy_id'] == buoy_id].sort_values('time').set_index('time')
    hs = buoy_df['Hs'].copy()
    if qc_mask is not None:
        qc_buoy = qc_mask[qc_mask['buoy_id'] == buoy_id].sort_values('time').set_index('time')
        hs[qc_buoy['Hs'].isin([2, 3, 4])] = np.nan
    result = {}
    annual_max = extract_annual_maxima(hs)
    result['annual_max'] = annual_max
    n_years = len(annual_max)
    result['n_years'] = n_years
    if n_years < 10:
        result['warning'] = f"仅{n_years}年数据，不足10年，仅输出频率直方图"
        if n_years > 0:
            result['histogram_fig'] = plot_histogram(annual_max.values)
        return result
    data = annual_max.values
    fits = []
    try:
        gumbel_fit = fit_gumbel(data)
        gumbel_fit['ks_test'] = ks_test(data, gumbel_fit)
        fits.append(gumbel_fit)
    except Exception as e:
        result['gumbel_error'] = str(e)
    try:
        weibull_fit = fit_weibull_3p(data)
        weibull_fit['ks_test'] = ks_test(data, weibull_fit)
        fits.append(weibull_fit)
    except Exception as e:
        result['weibull_error'] = str(e)
    result['fits'] = fits
    return_levels = {}
    for fit in fits:
        dist_name = fit['distribution']
        return_levels[dist_name] = {}
        for rp in RETURN_PERIODS:
            level = compute_return_level(fit, rp)
            try:
                ci = profile_likelihood_ci(data, fit, rp)
            except Exception:
                ci = (level * 0.8, level * 1.2)
            return_levels[dist_name][rp] = {'value': level, 'ci_lower': ci[0], 'ci_upper': ci[1]}
    result['return_levels'] = return_levels
    if fits:
        result['return_curve_fig'] = plot_return_curve(data, fits)
        result['qq_figs'] = {}
        for fit in fits:
            result['qq_figs'][fit['distribution']] = plot_qq(data, fit)
    return result
