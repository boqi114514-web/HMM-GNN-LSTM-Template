"""Portable, offline configuration. Importing this module never writes files.

The command-line entry point sets HMM_TEMPLATE_* before importing the pipeline.
Pickle input is retained for compatibility: load only files you trust.
"""
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

PROJECT_DIR = str(Path(__file__).resolve().parents[1])
_root = Path(PROJECT_DIR)
_base = json.loads((_root / 'configs/base.json').read_text(encoding='utf-8'))
_config_path = os.environ.get('HMM_TEMPLATE_CONFIG')
if _config_path:
    _custom = json.loads(Path(_config_path).read_text(encoding='utf-8'))
    if not isinstance(_custom, dict) or set(_custom) - set(_base):
        raise ValueError('Configuration must be an object containing known base.json keys only')
    _base.update(_custom)
PROFILE = os.environ.get('HMM_TEMPLATE_PROFILE', 'reference')
if PROFILE not in ('reference', 'smoke'):
    raise ValueError('Profile must be reference or smoke')
if PROFILE == 'smoke':
    _base.update(GAT_EPOCHS=2, GAT_PATIENCE=2, GAT_N_SEEDS=1,
                 LSTM_EPOCHS=2, LSTM_PATIENCE=2, LSTM_N_SEEDS=1,
                 HMM_N_ITER=20)
for _name, _value in _base.items():
    if _name.endswith(('MONTHS', 'DIM', 'EPOCHS', 'PATIENCE', 'SEEDS', 'WINDOW', 'ITER', 'STATES')) or _name in (
            'TOP_K', 'GRAPH_TOP_K', 'LSTM_LOOKBACK', 'LSTM_HIDDEN', 'LSTM_BATCH_SIZE'):
        if isinstance(_value, bool) or not isinstance(_value, int) or _value <= 0:
            raise ValueError(f'{_name} must be a positive integer')
if _base['HMM_N_STATES'] != 4 or _base['HMM_COVARIANCE'] != 'full':
    raise ValueError('The base architecture is four-state full-covariance HMM')
if _base['STEP_MONTHS'] != _base['PREDICT_MONTHS']:
    raise ValueError('STEP_MONTHS must equal PREDICT_MONTHS: no overlapping or missing forecasts')
for _name in ('GAT_DROPOUT', 'LSTM_DROPOUT'):
    if not 0 <= _base[_name] < 1:
        raise ValueError(f'{_name} must lie in [0,1)')
SETTINGS = dict(_base)
globals().update(_base)

LOCAL_DATA_DIR = str(Path(os.environ.get('HMM_TEMPLATE_DATA_DIR', _root / 'data')).resolve())
LOCAL_DATA_RAW = str(Path(LOCAL_DATA_DIR) / 'raw')
LOCAL_DATA_PROCESSED = str(Path(LOCAL_DATA_DIR) / 'processed')
OUTPUT_DIR = str(Path(os.environ.get('HMM_TEMPLATE_OUTPUT_DIR', _root / 'results/run')).resolve())
LEVEL = 'l1'  # Compatibility name only. No dependency on a particular taxonomy.
SELECTED_INDICATORS = [
    'totprofit', 'nptocostexpense', 'netprofitexcl', 'netprofitincl',
    'ebittointerest', 'opercash', 'grossprofitmargin', 'con_op_pr_yoy',
    'con_op_rt_yoy', 'con_np_cagh', 'con_tp_cagh', 'con_op_pr_cagh',
    'con_tp_yoy', 'operrev',
]


def load_industry_monthly():
    from data_pipeline.industry_monthly import canonicalize_industry_monthly
    frame = pd.read_csv(Path(LOCAL_DATA_RAW) / 'ts_sw_industry_monthly.csv', dtype={'ts_code': str})
    if not {'pe', 'pb'}.issubset(frame):
        raise ValueError('Industry prices require pe and pb columns')
    frame = canonicalize_industry_monthly(frame)
    frame = frame.loc[~frame.ts_code.isin(SW_EXCLUDE)].copy()
    if frame.empty or frame.ts_code.nunique() < max(10, TOP_K):
        raise ValueError('Original GAT training requires at least ten industry nodes')
    frame['year'], frame['month'] = frame.date.dt.year, frame.date.dt.month
    return frame.sort_values(['ts_code', 'date']).reset_index(drop=True)


def load_prosperity_monthly():
    frame = pd.read_pickle(Path(LOCAL_DATA_PROCESSED) / 'prosperity_indicators_clean.pkl')
    required = {'l1_code', 'year', 'quarter', *SELECTED_INDICATORS}
    if not required.issubset(frame) or frame.duplicated(['l1_code', 'year', 'quarter']).any():
        raise ValueError('Quarterly fundamentals have missing fields or duplicate node/quarter keys')
    records = []
    for row in frame.to_dict('records'):
        year, quarter = int(row['year']), int(row['quarter'])
        if quarter not in (1, 2, 3, 4):
            raise ValueError('quarter must be in 1..4')
        start = pd.Period(f'{year}Q{quarter}', freq='Q').asfreq('M', 'end') + 1
        for month in pd.period_range(start, periods=3, freq='M'):
            records.append({'ts_code': str(row['l1_code']), 'year': month.year, 'month': month.month,
                            **{c: row[c] for c in SELECTED_INDICATORS}})
    # Retains the original next-quarter proxy, NOT verified announcement timing.
    return pd.DataFrame(records)


def load_tech_factors(include_market=False):
    if include_market:
        raise ValueError('Market-factor extensions are outside this base template')
    pv = pd.read_pickle(Path(LOCAL_DATA_PROCESSED) / 'price_volume_factors.pkl').copy()
    patterns = pd.read_pickle(Path(LOCAL_DATA_PROCESSED) / 'pattern_factors.pkl').copy()
    for frame in (pv, patterns):
        frame['date'] = pd.to_datetime(frame.date)
        frame['ts_code'] = frame.ts_code.astype(str)
        frame['date'] = frame.date.dt.to_period('M').dt.to_timestamp('M')
        if frame.duplicated(['ts_code', 'date']).any():
            raise ValueError('Technical data must have unique node/month keys')
    tech = pv.merge(patterns, on=['ts_code', 'date'], how='left', validate='one_to_one')
    for name in ('pattern_median', 'bullish_ratio', 'signal_strength'):
        if name not in tech:
            raise ValueError(f'Missing pattern feature: {name}')
        tech[name] = tech[name].fillna(0.)
    factor_cols = [c for c in tech if c not in ('ts_code', 'date')]
    if len(factor_cols) != 12:
        raise ValueError('Base LSTM expects exactly 9 price-volume + 3 pattern factors')
    return tech.sort_values(['ts_code', 'date']).reset_index(drop=True)


def get_industries(mkt=None):
    return sorted((load_industry_monthly() if mkt is None else mkt).ts_code.unique())


def load_macro_factors():
    frame = pd.read_csv(Path(LOCAL_DATA_RAW) / 'ts_macro_factors.csv')
    frame['date'] = pd.to_datetime(frame.date.astype(str))
    return frame


def load_csi300_monthly():
    frame = pd.read_csv(Path(LOCAL_DATA_RAW) / 'ts_csi300_monthly.csv')
    frame['date'] = pd.to_datetime(frame.date.astype(str))
    return frame


def zscore_cross_section(df, cols, group_col='date'):
    frame = df.copy()
    for col in cols:
        grouped = frame.groupby(group_col)[col]
        frame[col] = ((frame[col] - grouped.transform('mean')) / (grouped.transform('std') + 1e-8)).fillna(0.)
    return frame


def check_data():
    prices, fundamental, technical = load_industry_monthly(), load_prosperity_monthly(), load_tech_factors()
    macro, csi = load_macro_factors(), load_csi300_monthly()
    required = {'pmi_mfg', 'shibor_1y', 'shibor_1m', 'm1_yoy_pct', 'm2_yoy_pct', 'sf_inc_month'}
    if not required.issubset(macro) or 'close' not in csi:
        raise ValueError('Macro or market index data lacks required columns')
    months = pd.PeriodIndex(prices.date.dt.to_period('M').unique()).sort_values()
    if len(months) < TRAIN_MONTHS + VAL_MONTHS + 2:
        raise ValueError('Need training + validation + at least two additional months')
    if not months.equals(pd.period_range(months[0], months[-1], freq='M')):
        raise ValueError('Industry calendar has missing months')
    for code, rows in prices.groupby('ts_code'):
        if len(rows) != len(months):
            raise ValueError(f'Base template requires a fixed complete industry universe: {code}')
    price_keys = set(zip(prices.ts_code, prices.year, prices.month))
    fundamental_keys = set(zip(fundamental.ts_code, fundamental.year, fundamental.month))
    technical_keys = set(zip(technical.ts_code, technical.date.dt.year, technical.date.dt.month))
    for name, keys in [('fundamental', fundamental_keys), ('technical', technical_keys)]:
        missing = price_keys - keys
        if missing:
            raise ValueError(f'{name} coverage is incomplete: {len(missing)} missing node/month keys; '
                             f'first examples: {sorted(missing)[:3]}')
    return {'profile': PROFILE, 'industries': prices.ts_code.nunique(), 'months': len(months),
            'start': str(months[0]), 'end': str(months[-1]), 'technical_rows': len(technical),
            'fundamental_month_rows': len(fundamental), 'macro_rows': len(macro),
            'market_index_rows': len(csi), 'settings': SETTINGS,
            'historical_fundamental_publication_versions_verified': False}
