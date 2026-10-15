"""Generate an offline, entirely artificial data fixture for the base pipeline.

Nothing in this module reads market data, contacts a provider, or claims to
reproduce a real index. Legacy filenames are schema compatibility names only.
The fixture checks software execution, never investment performance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FUNDAMENTAL_COLUMNS = (
    'totprofit', 'nptocostexpense', 'netprofitexcl', 'netprofitincl',
    'ebittointerest', 'opercash', 'grossprofitmargin', 'con_op_pr_yoy',
    'con_op_rt_yoy', 'con_np_cagh', 'con_tp_cagh', 'con_op_pr_cagh',
    'con_tp_yoy', 'operrev',
)
TECHNICAL_COLUMNS = (
    'mom_1m', 'mom_3m', 'mom_6m', 'vol_1m', 'vol_3m', 'turnover_chg',
    'vol_price_corr', 'max_drawdown_1m', 'high_low_pos',
)
PATTERN_COLUMNS = ('pattern_median', 'bullish_ratio', 'signal_strength')


def _ohlcv(code: str, dates: pd.DatetimeIndex, returns: np.ndarray,
           rng: np.random.Generator) -> pd.DataFrame:
    """Artificial monthly bars; pct_chg is percent, not a decimal return."""
    closes = 1000. * np.cumprod(1. + returns)
    previous = np.r_[1000., closes[:-1]]
    opens = previous * (1. + rng.normal(0., .003, len(dates)))
    spread = np.abs(rng.normal(.025, .008, len(dates)))
    volume = np.exp(rng.normal(15., .25, len(dates)))
    return pd.DataFrame({
        'ts_code': code, 'date': dates.strftime('%Y-%m-%d'),
        'open': opens, 'high': np.maximum(opens, closes) * (1. + spread),
        'low': np.minimum(opens, closes) / (1. + spread), 'close': closes,
        'pre_close': previous, 'change': closes - previous,
        'pct_chg': returns * 100., 'vol': volume, 'amount': volume * closes,
    })


def generate_demo(root: Path, months: int = 78, assets: int = 12,
                  seed: int = 42) -> dict:
    """Write a new synthetic data root, rejecting any existing destination.

    ``root`` contains ``raw/``, ``processed/``, and ``synthetic_manifest.json``.
    Defaults cover January 2010 through June 2016. A synthetic 2009Q4
    fundamental record supports the first month under the base loader's
    next-quarter mapping. The minimum dimensions match the reference model's
    60 training + 12 validation months and ten-node GAT requirement.

    All factor values are generated or derived from current/past synthetic
    observations. The three pattern columns are *stand-ins*, not outputs of
    the historical pattern-matching pipeline. Synthetic fundamentals and
    macro values have no real publication dates or economic interpretation.
    """
    for name, value, minimum in (('months', months, 74), ('assets', assets, 10)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError(f'Refusing to overwrite existing demo directory: {root}')

    rng = np.random.default_rng(seed)
    dates = pd.period_range('2010-01', periods=months, freq='M').to_timestamp('M')
    codes = [f'SYNTH{i + 1:03d}' for i in range(assets)]

    # Persistent latent variables generate a correlated but wholly fictional
    # universe. No fitted parameters or real market values are imported.
    latent = np.zeros((months, 3))
    latent[0] = rng.normal(size=3)
    for i in range(1, months):
        latent[i] = .75 * latent[i - 1] + rng.normal(0., .55, 3)
    exposures = rng.normal(0., .6, (assets, 3))
    common = .006 + .013 * latent[:, 0] + rng.normal(0., .025, months)
    returns = np.clip(
        common[:, None] + .015 * latent @ exposures.T
        + rng.normal(0., .035, (months, assets)), -.35, .35,
    )

    macro = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'shibor_1m': 2.5 + .35 * latent[:, 1] + rng.normal(0., .1, months),
        'shibor_1y': 3.1 + .30 * latent[:, 2] + rng.normal(0., .1, months),
        'pmi_mfg': 50. + 1.2 * latent[:, 0] + rng.normal(0., .4, months),
        'm1_yoy_pct': 8. + 2. * latent[:, 1] + rng.normal(0., .7, months),
        'm2_yoy_pct': 10. + latent[:, 2] + rng.normal(0., .5, months),
        'sf_inc_month': np.exp(9. + .20 * latent[:, 0] + rng.normal(0., .1, months)),
    })
    benchmark = _ohlcv('SYNTH_BENCHMARK', dates, returns.mean(axis=1), rng)
    market_parts, technical_parts, pattern_parts = [], [], []
    for j, code in enumerate(codes):
        market = _ohlcv(code, dates, returns[:, j], rng)
        market['pe'] = np.exp(3. + .2 * latent[:, 0] + .1 * exposures[j, 0])
        market['pb'] = np.exp(.6 + .12 * latent[:, 1] + .1 * exposures[j, 1])
        market_parts.append(market)
        ret = pd.Series(returns[:, j])
        close, vol = market['close'], market['vol']
        rolling_high = close.rolling(6, min_periods=1).max()
        rolling_low = close.rolling(6, min_periods=1).min()
        # The original column names are preserved, but intra-month statistics
        # below are explicit mock proxies built from these generated bars.
        technical_parts.append(pd.DataFrame({
            'ts_code': code, 'date': dates.strftime('%Y-%m-%d'),
            'mom_1m': ret,
            'mom_3m': (1. + ret).rolling(3, min_periods=1).apply(np.prod, raw=True) - 1.,
            'mom_6m': (1. + ret).rolling(6, min_periods=1).apply(np.prod, raw=True) - 1.,
            'vol_1m': (market['high'] - market['low']) / close,
            'vol_3m': ret.rolling(3, min_periods=1).std(ddof=0),
            'turnover_chg': vol.pct_change(fill_method=None).fillna(0.),
            'vol_price_corr': ret.rolling(6, min_periods=2).corr(vol).fillna(0.),
            'max_drawdown_1m': market['low'] / market['high'] - 1.,
            'high_low_pos': ((close - rolling_low) / (rolling_high - rolling_low))
                            .replace([np.inf, -np.inf], np.nan).fillna(.5),
        }))
        pattern_parts.append(pd.DataFrame({
            'ts_code': code, 'date': dates.strftime('%Y-%m-%d'),
            'pattern_median': ret.rolling(6, min_periods=1).median(),
            'bullish_ratio': ret.gt(0.).rolling(6, min_periods=1).mean(),
            'signal_strength': ret.rolling(3, min_periods=1).mean().abs(),
        }))

    quarter_records = []
    quarters = pd.period_range('2009Q4', dates[-1].to_period('Q'), freq='Q')
    fundamental_loading = rng.normal(0., .5, (3, len(FUNDAMENTAL_COLUMNS)))
    for quarter in quarters:
        known_idx = np.flatnonzero(dates.to_period('Q') <= quarter)
        # The leading synthetic quarter has its own random observation; no
        # January values are used to manufacture a December observation.
        quarter_state = latent[known_idx[-1]] if len(known_idx) else rng.normal(size=3)
        for j, code in enumerate(codes):
            values = ((quarter_state + exposures[j]) @ fundamental_loading
                      + rng.normal(0., .3, len(FUNDAMENTAL_COLUMNS)))
            quarter_records.append({
                'l1_code': code, 'year': int(quarter.year), 'quarter': int(quarter.quarter),
                **dict(zip(FUNDAMENTAL_COLUMNS, values.tolist())),
            })

    frames = {
        'raw/ts_macro_factors.csv': macro,
        'raw/ts_csi300_monthly.csv': benchmark,
        'raw/ts_sw_industry_monthly.csv': pd.concat(market_parts, ignore_index=True),
        'processed/prosperity_indicators_clean.pkl': pd.DataFrame(quarter_records),
        'processed/price_volume_factors.pkl': pd.concat(technical_parts, ignore_index=True),
        'processed/pattern_factors.pkl': pd.concat(pattern_parts, ignore_index=True),
    }
    for name, frame in frames.items():
        numeric = frame.select_dtypes(include=np.number).to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f'Non-finite generated values in {name}')

    # No directory is created before argument checks and data construction.
    root.mkdir(parents=True, exist_ok=False)
    (root / 'raw').mkdir()
    (root / 'processed').mkdir()
    files = {}
    for name, frame in frames.items():
        destination = root / name
        if destination.suffix == '.csv':
            frame.to_csv(destination, index=False, encoding='utf-8', float_format='%.12g')
        else:
            frame.to_pickle(destination)
        files[name] = {
            'rows': len(frame), 'columns': frame.columns.tolist(),
            'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
    manifest = {
        'schema_version': 1, 'synthetic': True, 'contains_real_market_data': False,
        'purpose': 'Offline software smoke test only; returns have no investment meaning.',
        'generator': 'src/demo_data.py', 'seed': seed, 'months': months, 'assets': assets,
        'start': dates[0].strftime('%Y-%m-%d'), 'end': dates[-1].strftime('%Y-%m-%d'),
        'industry_codes': codes, 'benchmark_code': 'SYNTH_BENCHMARK',
        'fundamentals_start_quarter': '2009Q4',
        'date_format': 'YYYY-MM-DD', 'pct_chg_unit': 'percent',
        'technical_features': list(TECHNICAL_COLUMNS),
        'pattern_features': list(PATTERN_COLUMNS),
        'limitations': [
            'Legacy SW/CSI300 filenames do not identify real markets in this fixture.',
            'Macro and fundamental values are arbitrary synthetic units, not observations.',
            'Technical and pattern columns are mock proxies, not actual daily-factor calculations.',
            'Next-quarter fundamental mapping does not establish real announcement availability.',
            'Smoke-test scores must never be presented as historical backtest performance.',
        ],
        'files': files,
    }
    (root / 'synthetic_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',
    )
    return {'root': str(root), **manifest}
