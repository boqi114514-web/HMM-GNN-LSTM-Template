"""Portable repaired original HMM + industry GAT + 12-factor LSTM workflow.

This template excludes fund allocation, stock selection and daily overlays.
Use the repository CLI to set data/output/profile before importing config.
All evaluation is monthly industry-index research, not executable stock P&L.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

import config
from backtest import calc_metrics, run_topk_strategy, simple_rank_ensemble, validate_backtest_inputs

MACRO_COLS = ['pmi_chg', 'term_spread', 'm1_m2_spread', 'sf_yoy']
PRICE_COLS = ['mkt_mom_1m', 'mkt_mom_3m', 'mkt_mom_6m', 'mkt_vol_3m',
              'mkt_vol_6m', 'mkt_vol_ratio', 'mkt_bias_6m', 'mkt_pos_12m']


def monthly_unique(frame, end):
    """Use the last observation in a month; retain calendar gaps explicitly."""
    if frame.empty or 'date' not in frame:
        raise ValueError('Monthly input is empty or lacks date')
    frame = frame.copy()
    frame['date'] = pd.to_datetime(frame['date'])
    frame = frame.sort_values('date', kind='stable')
    frame['ym'] = frame['date'].dt.to_period('M')
    frame = frame.drop_duplicates('ym', keep='last').set_index('ym')
    frame = frame.loc[frame.index <= pd.Timestamp(end).to_period('M')]
    if frame.empty:
        raise ValueError('Monthly input has no rows on or before end')
    frame = frame.reindex(pd.period_range(frame.index.min(), frame.index.max(), freq='M'))
    frame['date'] = frame.index.to_timestamp('M')
    return frame


def build_original_features(macro, csi, end):
    """Four macro and eight CSI300 features, with conservative macro lags.

Macro files without publication timestamps use a one-month lag for PMI,
money supply and financing changes. This assumption is not point-in-time
release-vintage verification and must be documented for each imported dataset.
"""
    macro = monthly_unique(macro, end)
    csi = monthly_unique(csi[['date', 'close']], end)
    macro['term_spread'] = macro['shibor_1y'] - macro['shibor_1m']
    macro['m1_m2_spread'] = macro['m1_yoy_pct'] - macro['m2_yoy_pct']
    macro['pmi_chg'] = macro['pmi_mfg'].diff()
    sf12 = macro['sf_inc_month'].rolling(12).sum()
    macro['sf_yoy'] = sf12 / sf12.shift(12) - 1
    lagged = ['pmi_chg', 'm1_m2_spread', 'sf_yoy']
    macro[lagged] = macro[lagged].shift(1)
    csi['ret'] = csi.close.pct_change(fill_method=None)
    for n in (1, 3, 6):
        csi[f'mkt_mom_{n}m'] = csi.close.pct_change(n, fill_method=None)
    for n in (3, 6):
        csi[f'mkt_vol_{n}m'] = csi.ret.rolling(n).std()
    csi['mkt_vol_ratio'] = csi.mkt_vol_3m / (csi.mkt_vol_6m + 1e-8)
    csi['mkt_bias_6m'] = csi.close / (csi.close.rolling(6).mean() + 1e-8) - 1
    lo, hi = csi.close.rolling(12).min(), csi.close.rolling(12).max()
    csi['mkt_pos_12m'] = (csi.close - lo) / (hi - lo + 1e-8)
    frame = csi[['date'] + PRICE_COLS].join(macro[MACRO_COLS], how='inner')
    cols = MACRO_COLS + PRICE_COLS
    frame[cols] = frame[cols].replace([np.inf, -np.inf], np.nan).ffill()
    return frame.dropna(subset=cols, how='all').reset_index(drop=True), cols


def _hmm_settings():
    return {'states': int(getattr(config, 'HMM_N_STATES', 4)),
            'covariance': getattr(config, 'HMM_COVARIANCE', getattr(config, 'HMM_COVARIANCE_TYPE', 'full')),
            'window': int(getattr(config, 'HMM_TRAIN_WINDOW', getattr(config, 'HMM_WINDOW', 60))),
            'iterations': int(getattr(config, 'HMM_N_ITER', getattr(config, 'HMM_ITERATIONS', 200))),
            'seed': int(getattr(config, 'RANDOM_SEED', 42))}


def rolling_original_hmm(features, cols, window=None):
    """Four full-covariance states; refit/scaling use only preceding months."""
    settings = _hmm_settings()
    window = settings['window'] if window is None else int(window)
    if settings['states'] != 4 or settings['covariance'] != 'full':
        raise ValueError('The original template requires four full-covariance HMM states')
    if window < 4 or settings['iterations'] < 1:
        raise ValueError('Invalid HMM window or iteration count')
    if len(features) <= window or 'mkt_mom_3m' not in cols:
        raise ValueError('Insufficient HMM history or missing mkt_mom_3m')
    features = features.copy()
    features['date'] = pd.to_datetime(features['date'])
    periods = pd.PeriodIndex(features['date'].dt.to_period('M'))
    if periods.has_duplicates or not periods.equals(pd.period_range(periods.min(), periods.max(), freq='M')):
        raise ValueError('HMM features must contain ordered consecutive calendar months')
    records = []
    logger = logging.getLogger('hmmlearn.base')
    original_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        for t in range(window, len(features)):
            past = features.iloc[t-window:t][cols].to_numpy(float)
            current = features.iloc[t:t+1][cols].to_numpy(float)
            past = np.where(np.isfinite(past), past, np.nan)
            current = np.where(np.isfinite(current), current, np.nan)
            means = pd.DataFrame(past).mean().fillna(0).to_numpy()
            past = np.where(np.isnan(past), means, past)
            current = np.where(np.isnan(current), means, current)
            scaler = StandardScaler().fit(past)
            x, now = scaler.transform(past), scaler.transform(current)
            failure, converged = '', False
            try:
                model = GaussianHMM(n_components=4, covariance_type='full',
                                    n_iter=settings['iterations'], random_state=settings['seed'])
                model.fit(x)
                labels = model.predict(x)
                mom_index = cols.index('mkt_mom_3m')
                order = sorted(range(4), key=lambda state: x[labels == state, mom_index].mean()
                               if (labels == state).any() else 0.0)
                mapping = {old: new for new, old in enumerate(order)}
                observed = np.vstack([x, now])
                state = mapping[model.predict(observed)[-1]]
                raw_prob = model.predict_proba(observed)[-1]
                prob = np.array([raw_prob[old] for old in order])
                if not np.isfinite(prob).all() or not np.isclose(prob.sum(), 1):
                    raise ValueError('Invalid state probabilities')
                converged = bool(model.monitor_.converged)
            except (ValueError, np.linalg.LinAlgError) as exc:
                state, prob = -1, np.full(4, .25)
                failure = f'{type(exc).__name__}: {exc}'
            records.append({'date': features.iloc[t]['date'], 'regime': state,
                            'converged': converged, 'failure': failure,
                            **{f'regime_prob_{i}': prob[i] for i in range(4)}})
    finally:
        logger.setLevel(original_level)
    result = pd.DataFrame(records)
    print(f'HMM: {len(result)} months; failed fits={(result.regime == -1).sum()}', flush=True)
    return result


def original_regime_ensemble(gnn, lstm, regimes, lookback=12):
    """Use matured, same-state historical Spearman IC; no LSTM weight floor."""
    if lookback < 1:
        raise ValueError('IC lookback must be positive')
    frame = simple_rank_ensemble(gnn, lstm, .5, .5)
    states = regimes[['date', 'regime']].copy()
    states['date'] = pd.to_datetime(states['date'])
    frame = frame.merge(states, on='date', how='left', validate='many_to_one')
    if frame.regime.isna().any() or not frame.regime.isin(range(-1, 4)).all():
        raise ValueError('HMM must cover every prediction month with state -1 or 0..3')
    earliest = frame['date'] + pd.offsets.MonthEnd(1)
    if 'label_available_at' not in frame:
        frame['label_available_at'] = earliest
    else:
        frame['label_available_at'] = pd.to_datetime(frame['label_available_at'])
        if frame['label_available_at'].isna().any() or (frame['label_available_at'] < earliest).any():
            raise ValueError('Forward labels cannot mature before the next month end')
    history = {state: [] for state in range(-1, 4)}
    pending, result = [], []
    for date, month in frame.sort_values(['date', 'ts_code']).groupby('date', sort=True):
        still_pending = []
        for ready, old_state, ics in pending:
            if ready <= date:
                history[old_state].append(ics)
            else:
                still_pending.append((ready, old_state, ics))
        pending = still_pending
        month = month.copy()
        state = int(month.regime.iloc[0])
        past = history[state][-lookback:]
        weights = np.maximum(0, np.mean(past, axis=0)) if past else np.array([.5, .5])
        weights = weights / weights.sum() if weights.sum() >= 1e-8 else np.array([.5, .5])
        month['w_gnn'], month['w_lstm'] = weights
        month['rank_ensemble'] = weights[0] * month.rank_gnn + weights[1] * month.rank_lstm
        month['pred_ensemble'] = -month.rank_ensemble
        month['matured_same_state_ic_observations'] = len(past)
        result.append(month)
        labeled = month.dropna(subset=['actual_ret'])
        if len(labeled) >= 5:
            ics = [spearmanr(labeled[column], labeled.actual_ret).statistic
                   for column in ('pred_gnn', 'pred_lstm_b')]
            pending.append((labeled.label_available_at.max(), state,
                            [float(value) if np.isfinite(value) else 0.0 for value in ics]))
    return pd.concat(result, ignore_index=True)


def input_hashes():
    """Portable relative manifest keys, with no paths, credentials or old caches."""
    root = Path(config.PROJECT_DIR)
    files = {f'src/{path.relative_to(root / "src").as_posix()}': path
             for path in sorted((root / 'src').rglob('*.py'))}
    for relative in ('run.py', 'configs/base.json'):
        if (root / relative).is_file():
            files[relative] = root / relative
    files.update({f'data/raw/{name}': Path(config.LOCAL_DATA_RAW) / name for name in
                  ('ts_macro_factors.csv', 'ts_csi300_monthly.csv', 'ts_sw_industry_monthly.csv')})
    files.update({f'data/processed/{name}': Path(config.LOCAL_DATA_PROCESSED) / name for name in
                  ('prosperity_indicators_clean.pkl', 'price_volume_factors.pkl', 'pattern_factors.pkl')})
    return {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()}


def backtest(output):
    """Evaluate only this run's branches and the original same-state IC mix."""
    output = Path(output)
    market = config.load_industry_monthly()
    gnn = pd.read_pickle(output / 'predictions_gnn.pkl')
    lstm = pd.read_pickle(output / 'predictions_lstm_b.pkl')
    coverage = validate_backtest_inputs(gnn, lstm, market, top_k=config.TOP_K)
    regimes = pd.read_pickle(output / 'regime_labels.pkl')
    ensemble = original_regime_ensemble(gnn, lstm, regimes)
    ensemble.to_pickle(output / 'predictions_ensemble.pkl')
    ensemble.to_csv(output / 'predictions_ensemble.csv', index=False, encoding='utf-8-sig')
    ensemble.groupby('date')[['regime', 'w_gnn', 'w_lstm',
                              'matured_same_state_ic_observations']].first().to_csv(
        output / 'monthly_weights.csv', encoding='utf-8-sig')
    fixed = simple_rank_ensemble(gnn, lstm, .5, .5)
    returns = {}
    for name, frame, score in [('gnn', gnn, 'pred_gnn'), ('lstm', lstm, 'pred_lstm_b'),
                                ('fixed_50_50', fixed, 'pred_ensemble'),
                                ('hmm_gnn_lstm', ensemble, 'pred_ensemble')]:
        returns[name], holdings = run_topk_strategy(frame, score, k=config.TOP_K, inertia=.2)
        holdings.to_csv(output / f'holdings_{name}.csv', index=False, encoding='utf-8-sig')
    # Equal-weight benchmark on exactly the prediction universe and signal months.
    returns['industry_equal_weight'] = gnn.groupby('date').actual_ret.mean().reindex(returns['hmm_gnn_lstm'].index)
    expected = coverage[:-1]
    for name, values in returns.items():
        if not values.index.to_period('M').equals(expected) or values.isna().any():
            raise ValueError(f'{name}: incomplete or unequal holding-month coverage')
    monthly = pd.DataFrame(returns).sort_index()
    monthly.index.name = 'signal_date'
    held = monthly.copy()
    held.index += pd.offsets.MonthEnd(1)
    held.index.name = 'holding_month'
    monthly.insert(0, 'holding_month', held.index)
    monthly.to_csv(output / 'monthly_returns.csv', encoding='utf-8-sig')
    (1 + held).cumprod().to_csv(output / 'nav.csv', encoding='utf-8-sig')
    annual, summaries = [], []
    for strategy in held:
        summaries.append({'strategy': strategy, **calc_metrics(held[strategy], rf=config.RF_ANNUAL)})
        for year, values in held[strategy].groupby(held.index.year):
            annual.append({'strategy': strategy, 'year': year, 'n_months': len(values),
                           'start': values.index.min(), 'end': values.index.max(),
                           'return': (1 + values).prod() - 1})
    pd.DataFrame(annual).to_csv(output / 'annual_returns.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(summaries).to_csv(output / 'summary.csv', index=False, encoding='utf-8-sig')
    return {'signal_start': str(coverage[0]), 'signal_end': str(coverage[-1]),
            'holding_start': str(held.index.min().date()), 'holding_end': str(held.index.max().date()),
            'n_realized_months': len(held), 'hmm_failure_months': int((regimes.regime == -1).sum()),
            'hmm_nonconverged_months': int((~regimes.converged.astype(bool)).sum())}


def main(output_dir=None):
    """Write one self-contained new run; never overwrite existing artifacts."""
    output = Path(output_dir if output_dir is not None else config.OUTPUT_DIR)
    if output.exists():
        raise FileExistsError('Output must be a new directory; previous runs are preserved')
    before = input_hashes()
    output.mkdir(parents=True, exist_ok=True)
    print('[1/4] Original four-state macro HMM', flush=True)
    end = config.load_industry_monthly().date.max()
    features, columns = build_original_features(config.load_macro_factors(), config.load_csi300_monthly(), end)
    features.to_csv(output / 'hmm_features.csv', index=False, encoding='utf-8-sig')
    regimes = rolling_original_hmm(features, columns)
    regimes.to_pickle(output / 'regime_labels.pkl')
    regimes.to_csv(output / 'regime_labels.csv', index=False, encoding='utf-8-sig')
    print('[2/4] GAT with state one-hot features', flush=True)
    from s1_gnn_train import main as train_gnn
    train_gnn(regime_df=regimes, output_dir=str(output))
    print('[3/4] Original twelve-factor LSTM', flush=True)
    from s2_lstm_b_train import main as train_lstm
    train_lstm(use_market_factors=False, output_dir=str(output))
    print('[4/4] Matured same-state IC fusion and matched industry evaluation', flush=True)
    details = backtest(output)
    if before != input_hashes():
        raise RuntimeError('Source or inputs changed during run; outputs are not a verified completed run')
    manifest = {
        'architecture_reference': 'Regime-GNN-LSTM repaired original, historical reference 818423c',
        'source_snapshot': '33edeff9dbba6f7c3913b48790fbc938b61c451c',
        'architecture': 'macro HMM(4,full) + GLASSO/GAT + industry12 LSTM + matured same-state IC',
        'evaluation': 'monthly industry-index close-to-close research, not a stock execution ledger',
        'profile': getattr(config, 'PROFILE', 'reference'),
        'settings': getattr(config, 'SETTINGS', {}),
        'hmm': _hmm_settings(), 'top_k_industries': config.TOP_K, 'inertia': .2,
        'weighting': {'same_state_ic_observations': 12, 'lstm_floor': None},
        'repairs': ['canonical calendar months and close-derived decimal returns',
                    'forward labels with train/validation label cutoffs',
                    'retain partial final prediction window and unlabeled latest signal',
                    'SHIBOR 1y minus 1m; one-month lag for macro data without publication timestamps',
                    'do not cross-sectionally normalize shared regime one-hot columns',
                    'explicit next-month label maturity before historical IC use',
                    'score-only Top-K selection; missing selected realized labels fail closed',
                    'include initial capital in drawdown; deterministic score-tie ordering'],
        'known_data_limitations': ['macro publication vintages are not verified by a one-month lag',
                                   'quarterly fundamental availability follows the configured data contract',
                                   'provided universe and factor history require independent point-in-time review'],
        'input_sha256': before,
        'prediction_sha256': {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in
                              ('predictions_gnn.pkl', 'predictions_lstm_b.pkl', 'regime_labels.pkl')},
        **details,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Complete: {output}', flush=True)
    return manifest


if __name__ == '__main__':
    main()
