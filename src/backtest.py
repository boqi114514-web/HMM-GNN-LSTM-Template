"""Monthly industry-index research accounting, not a stock execution ledger.

Scores at month t are evaluated against close-to-close index returns in t+1.
No fees, lots, suspended stocks, transaction prices or next-open fills are
modelled.  These functions deliberately have no filesystem/import side effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _prediction_frame(frame, score, name):
    required = {'ts_code', 'date', 'actual_ret', score}
    if not required.issubset(frame.columns):
        raise ValueError(f'{name}: missing columns {sorted(required - set(frame.columns))}')
    frame = frame.copy()
    frame['date'] = pd.to_datetime(frame['date'])
    if frame.empty or frame.duplicated(['ts_code', 'date']).any():
        raise ValueError(f'{name}: empty or duplicate industry/signal-month predictions')
    if not frame['date'].eq(frame['date'].dt.to_period('M').dt.to_timestamp('M')).all():
        raise ValueError(f'{name}: dates must be canonical calendar month ends')
    if not np.isfinite(frame[score].to_numpy(float)).all():
        raise ValueError(f'{name}: nonfinite prediction score')
    labels = frame['actual_ret'].dropna().to_numpy(float)
    if not np.isfinite(labels).all() or (labels < -1).any():
        raise ValueError(f'{name}: invalid decimal return labels')
    return frame


def validate_backtest_inputs(gnn_df, lstm_df, mkt, top_k=5, as_of=None):
    """Require matching node/month universes and next-calendar-month labels."""
    required = {'ts_code', 'date', 'ret'}
    if not required.issubset(mkt.columns):
        raise ValueError(f'Market missing columns: {sorted(required - set(mkt.columns))}')
    market = mkt.copy().sort_values(['ts_code', 'date'])
    market['date'] = pd.to_datetime(market['date'])
    if market.empty or market.duplicated(['ts_code', 'date']).any():
        raise ValueError('Market is empty or contains duplicate industry/month rows')
    next_date = market.groupby('ts_code')['date'].shift(-1)
    consecutive = next_date.dt.to_period('M').astype('int64').sub(
        market['date'].dt.to_period('M').astype('int64')).eq(1)
    market['fwd_ret'] = market.groupby('ts_code')['ret'].shift(-1).where(consecutive)
    labels = market[['ts_code', 'date', 'fwd_ret']]
    last_month = market['date'].max().to_period('M')
    now = pd.Timestamp.now(tz='Asia/Shanghai') if as_of is None else pd.Timestamp(as_of)
    if last_month >= now.to_period('M'):
        raise ValueError(f'Market includes unfinished month {last_month}')

    frames = []
    for name, frame, score in [('GNN', gnn_df, 'pred_gnn'),
                               ('LSTM', lstm_df, 'pred_lstm_b')]:
        frame = _prediction_frame(frame, score, name)
        periods = pd.PeriodIndex(frame['date'].dt.to_period('M').unique()).sort_values()
        if periods.max() != last_month:
            raise ValueError(f'{name}: predictions do not reach last market month {last_month}')
        missing = pd.period_range(periods.min(), last_month, freq='M').difference(periods)
        if len(missing):
            raise ValueError(f'{name}: missing prediction months {list(map(str, missing))}')
        checked = frame.merge(labels, on=['ts_code', 'date'], how='left',
                              validate='one_to_one', indicator=True)
        if not checked['_merge'].eq('both').all():
            raise ValueError(f'{name}: predictions contain unknown market keys')
        if not checked['actual_ret'].notna().eq(checked['fwd_ret'].notna()).all():
            raise ValueError(f'{name}: label availability differs from next-month market return')
        present = checked['fwd_ret'].notna()
        if not np.allclose(checked.loc[present, 'actual_ret'],
                           checked.loc[present, 'fwd_ret'], atol=1e-6, rtol=1e-5):
            raise ValueError(f'{name}: labels are not next-calendar-month decimal returns')
        frames.append(frame)

    left = pd.MultiIndex.from_frame(frames[0][['ts_code', 'date']])
    right = pd.MultiIndex.from_frame(frames[1][['ts_code', 'date']])
    if len(left.difference(right)) or len(right.difference(left)):
        raise ValueError('GNN and LSTM must have identical industry/signal-month keys')
    expected = pd.period_range(frames[0]['date'].min().to_period('M'), last_month, freq='M')
    for frame in frames:
        counts = frame.groupby(frame['date'].dt.to_period('M'))['actual_ret'].count()
        if any(counts.get(period, 0) < top_k for period in expected[:-1]):
            raise ValueError('A realized month has fewer return labels than Top-K')
    if len(expected) < 2:
        raise ValueError('At least one realized and one pending signal month are required')
    return expected


def simple_rank_ensemble(gnn_df, lstm_df, w_gnn=0.5, w_lstm=0.5):
    """Combine two branches by within-month rank, without a branch-weight floor."""
    if not np.isfinite([w_gnn, w_lstm]).all() or min(w_gnn, w_lstm) < 0 or not np.isclose(w_gnn + w_lstm, 1):
        raise ValueError('Weights must be finite, nonnegative and sum to one')
    left = _prediction_frame(gnn_df, 'pred_gnn', 'GNN').drop(columns='pred_lstm_b', errors='ignore')
    right = _prediction_frame(lstm_df, 'pred_lstm_b', 'LSTM')
    merged = left.merge(right[['ts_code', 'date', 'pred_lstm_b']],
                        on=['ts_code', 'date'], validate='one_to_one')
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError('Both branches must have exactly the same prediction keys')
    merged['rank_gnn'] = merged.groupby('date')['pred_gnn'].rank(ascending=False)
    merged['rank_lstm'] = merged.groupby('date')['pred_lstm_b'].rank(ascending=False)
    merged['w_gnn'], merged['w_lstm'] = w_gnn, w_lstm
    merged['rank_ensemble'] = w_gnn * merged.rank_gnn + w_lstm * merged.rank_lstm
    merged['pred_ensemble'] = -merged['rank_ensemble']
    return merged.sort_values(['date', 'ts_code']).reset_index(drop=True)


def run_topk_strategy(pred_df, pred_col, k=5, inertia=0.0):
    """Equal-weight Top-K indices, with optional prior-holding score bonus.

Selection never filters on a future label. A selected missing realized label
fails loudly instead of replacing that index with an ex-post surviving one.
The final, fully unlabeled month is logged as pending, not included in returns.
"""
    if k < 1 or int(k) != k or not np.isfinite(inertia) or inertia < 0:
        raise ValueError('k must be a positive integer; inertia must be finite and nonnegative')
    frame = _prediction_frame(pred_df, pred_col, 'Strategy')
    records, holdings = [], []
    previous = set()
    last_date = frame['date'].max()
    for date, month in frame.sort_values(['date', 'ts_code']).groupby('date', sort=True):
        month = month.copy()
        if len(month) < k:
            raise ValueError(f'{date:%Y-%m}: fewer score candidates than Top-{k}')
        if inertia and previous:
            bonus = inertia * month[pred_col].std()
            bonus = float(bonus) if np.isfinite(bonus) else 0.0
            month.loc[month.ts_code.isin(previous), pred_col] += bonus
        selected = month.sort_values([pred_col, 'ts_code'], ascending=[False, True],
                                      kind='stable').head(k)
        pending = month['actual_ret'].isna().all()
        if pending and date != last_date:
            raise ValueError(f'{date:%Y-%m}: unlabeled non-final month')
        if not pending and selected['actual_ret'].isna().any():
            raise ValueError(f'{date:%Y-%m}: selected index has no realized label; refusing ex-post replacement')
        current = set(selected.ts_code)
        value = float(selected.actual_ret.mean()) if not pending else np.nan
        holdings.append({'date': date, 'holding_month': date + pd.offsets.MonthEnd(1),
                         'holdings': selected.ts_code.tolist(),
                         'turnover': 1 - len(current & previous) / k if previous else 1.0,
                         'port_ret': value, 'status': 'pending' if pending else 'realized'})
        if not pending:
            records.append({'date': date, 'ret': value})
        previous = current
    returns = pd.DataFrame(records)
    if returns.empty:
        return pd.Series(dtype=float, name='ret'), pd.DataFrame(holdings)
    return returns.set_index('date')['ret'], pd.DataFrame(holdings)


def calc_metrics(rets, rf=0.03):
    """Monthly metrics; maximum drawdown includes the initial unit of capital."""
    rets = pd.Series(rets).dropna()
    if rets.empty:
        return {}
    if not np.isfinite(rets.to_numpy(float)).all() or (rets < -1).any():
        raise ValueError('Returns must be finite decimals no smaller than -1')
    nav = (1 + rets).cumprod()
    total = float(nav.iloc[-1] - 1)
    annual = float((1 + total) ** (12 / len(rets)) - 1)
    volatility = float(rets.std() * np.sqrt(12)) if len(rets) > 1 else 0.0
    high_water = nav.cummax().clip(lower=1.0)
    return {'total_return': total, 'annual_return': annual,
            'annual_volatility': volatility,
            'sharpe_ratio': (annual - rf) / volatility if volatility > 0 else 0.0,
            'max_drawdown': float((nav / high_water - 1).min()),
            'win_rate': float((rets > 0).mean()), 'n_months': len(rets)}
