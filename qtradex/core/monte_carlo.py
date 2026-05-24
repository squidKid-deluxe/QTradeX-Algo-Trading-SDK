import multiprocessing as mp
import random
import os

import matplotlib.pyplot as plt
import numpy as np

import qtradex as qx
from qtradex.plot.utilities import unix_to_stamp
from qtradex.private.wallet import PaperWallet

_WORKER_BOT = None
_WORKER_DATA = None
_WORKER_WALLET = None
_WORKER_KWARGS = {}


def _worker_backtest(tune_dict):
    _WORKER_BOT.tune = tune_dict
    ret, raw, states = qx.backtest(
        _WORKER_BOT, _WORKER_DATA, _WORKER_WALLET.copy(),
        plot=False, return_states=True, **_WORKER_KWARGS,
    )
    pc = qx.indicators.fitness.percent_cheats(states)
    roi = _roi_curve(raw, _WORKER_DATA.asset, _WORKER_DATA.currency, pc)
    return raw["unix"], roi


def _perturb_scalar(value, minv, maxv, perturbation):
    span = maxv - minv
    delta = random.uniform(-perturbation, perturbation) * span
    new_val = value + delta
    new_val = max(float(minv), min(float(maxv), float(new_val)))
    return int(new_val) if isinstance(value, int) else new_val


def _perturb_array(value, minv, maxv, perturbation):
    span = maxv - minv
    delta = np.random.uniform(-perturbation, perturbation, size=value.shape) * span
    new_val = value + delta
    return np.clip(new_val, minv, maxv).astype(value.dtype)


def _parse_clamp(clamp):
    if len(clamp) >= 4:
        return clamp[0], clamp[2]
    elif len(clamp) == 3:
        return clamp[0], clamp[1]
    elif len(clamp) == 2:
        return clamp[0], clamp[1]
    return None, None


def perturb_tune(bot, perturbation=0.01):
    new_tune = {}
    items = list(bot.tune.items())

    if isinstance(bot.clamps, dict):
        for key, value in items:
            clamp = bot.clamps.get(key)
            if clamp is None:
                new_tune[key] = value
                continue
            minv, maxv = _parse_clamp(clamp)
            if minv is None:
                new_tune[key] = value
                continue
            new_tune[key] = (
                _perturb_array(value, minv, maxv, perturbation)
                if isinstance(value, np.ndarray)
                else _perturb_scalar(value, minv, maxv, perturbation)
            )
    else:
        for idx, (key, value) in enumerate(items):
            if idx < len(bot.clamps):
                minv, maxv = _parse_clamp(bot.clamps[idx])
                if minv is not None:
                    new_tune[key] = (
                        _perturb_array(value, minv, maxv, perturbation)
                        if isinstance(value, np.ndarray)
                        else _perturb_scalar(value, minv, maxv, perturbation)
                    )
                    continue
            new_tune[key] = value

    return new_tune


def simulate_random_trade(close, period=7, seed=42):
    rng = random.Random(seed)
    n = len(close)
    asset = 0.0
    currency = 1.0
    values = np.ones(n)
    for i in range(n):
        if i % period == 0:
            action = rng.choice(["buy", "sell", "nothing"])
            price = close[i]
            if action == "buy" and currency > 0:
                asset = currency / price
                currency = 0.0
            elif action == "sell" and asset > 0:
                currency = asset * price
                asset = 0.0
        values[i] = asset * close[i] + currency
    return values / values[0]


def _roi_curve(raw_states, asset, currency, pc=0):
    balances = raw_states["balances"]
    prices = raw_states["close"]
    n = len(balances)
    if n < 2:
        return np.ones(n)

    b0 = balances[0]
    a0 = b0[asset]
    c0 = b0[currency]
    p0 = prices[0]
    curve = np.ones(n)
    for i in range(1, n):
        bi = balances[i]
        ai, ci, pi = bi[asset], bi[currency], prices[i]
        r_assets = (ci + ai * pi) / (c0 + a0 * p0)
        r_currency = (ai + ci / pi) / (a0 + c0 / p0)
        r_static_a = (c0 + a0 * pi) / (c0 + a0 * p0)
        r_static_c = (a0 + c0 / pi) / (a0 + c0 / p0)
        curve[i] = ((r_assets * r_currency) ** 0.5 / (r_static_a * r_static_c) ** 0.5) * (1 + pc / 100)
    return curve


def monte_carlo(bot, data, wallet=None, iterations=50, perturbation=0.01, plot=True, block=True, **kwargs):
    if wallet is None:
        wallet = PaperWallet({data.asset: 0, data.currency: 1})

    orig_tune = bot.tune.copy()

    # --- Baseline ---
    baseline = qx.backtest(
        bot, data, wallet.copy(),
        plot=False, return_states=True, **kwargs,
    )
    baseline_ret, baseline_raw, baseline_states = baseline
    baseline_t = unix_to_stamp(baseline_raw["unix"])
    pc_baseline = qx.indicators.fitness.percent_cheats(baseline_states)
    baseline_vals = _roi_curve(baseline_raw, data.asset, data.currency, pc_baseline)
    baseline_final = baseline_vals[-1]
    close = baseline_raw["close"]
    buy_hold = close / close[0]
    sell_hold = np.ones_like(close)
    random_trade = simulate_random_trade(close)

    # --- Pre-generate perturbed tunes ---
    tunes = [perturb_tune(bot, perturbation) for _ in range(iterations)]

    # --- Set up parallel workers ---
    global _WORKER_BOT, _WORKER_DATA, _WORKER_WALLET, _WORKER_KWARGS
    _WORKER_BOT = bot
    _WORKER_DATA = data
    _WORKER_WALLET = wallet
    _WORKER_KWARGS = {
        k: kwargs[k] for k in ("range_periods", "fine_data", "always_trade")
        if k in kwargs
    }

    results = []
    try:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=os.cpu_count()) as pool:
            total = len(tunes)
            chunksize = max(1, total // (os.cpu_count() * 4))
            for i, r in enumerate(
                pool.imap_unordered(_worker_backtest, tunes, chunksize=chunksize), 1
            ):
                results.append(r)
                print(f"\rMC: {i}/{total} ({100 * i // total}%)", end="", flush=True)
            print()
    except Exception as e:
        print(f"\nParallel execution failed ({e}), falling back to serial...")
        tunes = tunes[:150]
        results = []
        total = len(tunes)
        for i, t in enumerate(tunes, 1):
            results.append(_worker_backtest(t))
            print(f"\rMC: {i}/{total} ({100 * i // total}%)", end="", flush=True)
        print()

    runs = len(results)

    # --- Per-tick statistics (align all curves to baseline timeline) ---
    baseline_unix = baseline_raw["unix"]
    all_aligned = []
    for unix, roi in results:
        all_aligned.append(np.interp(baseline_unix, unix, roi, left=np.nan, right=np.nan))

    all_aligned = np.array(all_aligned)
    mean_curve = np.exp(np.nanmean(np.log(all_aligned), axis=0))
    p5_curve = np.nanpercentile(all_aligned, 5, axis=0)
    p95_curve = np.nanpercentile(all_aligned, 95, axis=0)

    finals = all_aligned[:, -1]
    mean_f = float(np.exp(np.nanmean(np.log(finals))))
    std_f = float(np.nanstd(finals))
    p5 = float(np.nanpercentile(finals, 5))
    p95 = float(np.nanpercentile(finals, 95))

    log_all = np.log(all_aligned)
    log_mean_center = np.nanmean(log_all, axis=0)
    log_worst = np.nanmin(log_all, axis=0)
    log_best = np.nanmax(log_all, axis=0)
    denom = log_best - log_worst
    denom = np.where(denom == 0, 1, denom)
    skew_2d = float(np.nanmean((log_mean_center - log_worst) / denom))

    # --- Plot ---
    if plot:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_yscale("log")
        ax.set_title(
            f"Monte Carlo — {runs} runs,  ±{perturbation * 100:.0f}% perturbation"
        )
        ax.set_ylabel("Portfolio Value (normalized)")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
        ax.step(baseline_t, buy_hold, color="white", linestyle="--", linewidth=2,
                alpha=0.8, label=f"Buy & hold (final: {buy_hold[-1]:.4f})", where="post")
        ax.step(baseline_t, sell_hold, color="orange", linestyle="--", linewidth=2,
                alpha=0.8, label=f"Sell & hold (final: {sell_hold[-1]:.4f})", where="post")
        ax.step(baseline_t, random_trade, color="magenta", linestyle=":", linewidth=2,
                alpha=0.8, label=f"Random trade/every 7 (final: {random_trade[-1]:.4f})",
                where="post")
        ax.step(
            baseline_t, baseline_vals, color="cyan", linewidth=1.5,
            label=f"Baseline (final: {baseline_final:.4f})", where="post",
        )

        for unix, roi in results:
            ax.step(unix_to_stamp(unix), roi, color="gray",
                    alpha=0.12, linewidth=0.5, where="post")

        ax.step(baseline_t, mean_curve, color="yellow", linewidth=1.5, alpha=0.8,
                label=f"Mean (final: {mean_f:.4f})", where="post")
        ax.step(baseline_t, p5_curve, color="red", linewidth=1, alpha=0.6,
                label=f"P5 (final: {p5:.4f})", where="post")
        ax.step(baseline_t, p95_curve, color="lime", linewidth=1, alpha=0.6,
                label=f"P95 (final: {p95:.4f})", where="post")
        ax.step(baseline_t, baseline_vals, color="cyan", linewidth=1.5, where="post")

        ax.legend(loc="best")
        plt.tight_layout()
        print(f"Baseline final: {baseline_final:.4f}")
        print(f"MC Mean:        {mean_f:.4f}  (\u03c3={std_f:.4f})")
        print(f"  P5: {p5:.4f}   P95: {p95:.4f}")
        print(f"  2D Skew:      {skew_2d:.4f}  (0=worst-bound, 0.5=centered, 1=best-bound)")
        print(f"Worse than baseline: {(finals <= baseline_final).mean() * 100:.1f}%")
        plt.show(block=block)

    bot.tune = orig_tune

    return {
        "baseline_final": baseline_final,
        "baseline_ret": baseline_ret,
        "baseline_vals": baseline_vals,
        "mean_f": mean_f,
        "std_f": std_f,
        "p5": p5,
        "p95": p95,
        "skew_2d": skew_2d,
        "mean_curve": mean_curve,
        "p5_curve": p5_curve,
        "p95_curve": p95_curve,
        "timestamps": baseline_t,
        "all_aligned": all_aligned,
        "mc_results": [{"unix": u, "roi": r} for u, r in results],
    }
