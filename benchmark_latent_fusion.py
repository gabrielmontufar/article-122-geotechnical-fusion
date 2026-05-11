import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
TAB = OUT / "table_images"
DATA = OUT / "data"
for folder in (FIG, TAB, DATA):
    folder.mkdir(parents=True, exist_ok=True)


def exp_cov(xa, xb, scale, variance=1.0, nugget=0.0):
    xa = np.asarray(xa)[:, None]
    xb = np.asarray(xb)[None, :]
    mat = variance * np.exp(-np.abs(xa - xb) / scale)
    if xa.shape[0] == xb.shape[1] and np.allclose(xa.ravel(), xb.ravel()):
        mat = mat + nugget * np.eye(xa.shape[0])
    return mat


def smooth_profile(y, width=9):
    kernel = np.hanning(width)
    kernel = kernel / kernel.sum()
    pad = width // 2
    padded = np.pad(y, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def sample_fields(rng, z, params):
    e = rng.multivariate_normal(
        np.zeros(len(z)),
        exp_cov(z, z, params["ell_common"], params["sigma_common"] ** 2, 1e-8),
    )
    rx = rng.multivariate_normal(
        np.zeros(len(z)),
        exp_cov(z, z, params["ell_residual"], params["sigma_residual"] ** 2, 1e-8),
    )
    rz = rng.multivariate_normal(
        np.zeros(len(z)),
        exp_cov(z, z, params["ell_geo_noise"], params["sigma_geo_noise"] ** 2, 1e-8),
    )
    x = e + rx
    zgeo = params["lambda_geo"] * e + rz
    zgeo = smooth_profile(zgeo, width=params["support_width"])
    return e, x, zgeo


def interp_at(z_grid, y_grid, z_obs):
    return np.interp(z_obs, z_grid, y_grid)


def profile_nll(y, cov):
    cov = cov + 1e-8 * np.eye(cov.shape[0])
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return np.inf
    alpha = np.linalg.solve(cov, y)
    return 0.5 * (logdet + y @ alpha + len(y) * np.log(2 * np.pi))


def gp_condition(train_x, train_y, pred_x, scale, variance, noise_var):
    k_tt = exp_cov(train_x, train_x, scale, variance, noise_var)
    k_pt = exp_cov(pred_x, train_x, scale, variance)
    k_pp = exp_cov(pred_x, pred_x, scale, variance)
    alpha = np.linalg.solve(k_tt, train_y)
    mean = k_pt @ alpha
    cov = k_pp - k_pt @ np.linalg.solve(k_tt, k_pt.T)
    var = np.clip(np.diag(cov), 1e-8, None)
    return mean, var


def estimate_direct_ml(z_direct, y_direct, length_grid, noise):
    y = y_direct - np.mean(y_direct)
    var = max(np.var(y), 1e-4)
    nll = []
    for ell in length_grid:
        cov = exp_cov(z_direct, z_direct, ell, var, noise**2)
        nll.append(profile_nll(y, cov))
    nll = np.array(nll)
    ell_hat = float(length_grid[np.argmin(nll)])
    w = np.exp(-0.5 * (nll - nll.min()))
    w = w / w.sum()
    return ell_hat, w


def estimate_proxy_length(z_geo, y_geo, length_grid, noise):
    y = y_geo - np.mean(y_geo)
    var = max(np.var(y), 1e-4)
    nll = []
    for ell in length_grid:
        cov = exp_cov(z_geo, z_geo, ell, var, noise**2)
        nll.append(profile_nll(y, cov))
    nll = np.array(nll)
    ell_hat = float(length_grid[np.argmin(nll)])
    w = np.exp(-0.5 * (nll - nll.min()))
    w = w / w.sum()
    return ell_hat, w


def empirical_corr_length(z_direct, y_direct):
    pairs_h = []
    pairs_g = []
    for i in range(len(y_direct)):
        for j in range(i + 1, len(y_direct)):
            pairs_h.append(abs(z_direct[i] - z_direct[j]))
            pairs_g.append(0.5 * (y_direct[i] - y_direct[j]) ** 2)
    h = np.array(pairs_h)
    g = np.array(pairs_g)
    sill = max(np.var(y_direct), 1e-6)
    target = 0.632 * sill
    idx = np.argmin(np.abs(g - target))
    return float(np.clip(h[idx], 0.3, 8.0))


def regression_assisted(z_direct, y_direct, z_geo, y_geo, z_hold, x_hold, length_grid, noise):
    geo_at_direct = interp_at(z_geo, y_geo, z_direct)
    a, b = np.polyfit(geo_at_direct, y_direct, 1)
    residual = y_direct - (a * geo_at_direct + b)
    ell_hat, _ = estimate_direct_ml(z_direct, residual, length_grid, noise)
    residual_mean, residual_var = gp_condition(
        z_direct, residual, z_hold, ell_hat, max(np.var(residual), 1e-4), noise**2
    )
    pred = a * interp_at(z_geo, y_geo, z_hold) + b + residual_mean
    rmse = float(np.sqrt(np.mean((x_hold - pred) ** 2)))
    return ell_hat, rmse, pred, float(a), float(b)


def fusion_grid(z_direct, y_direct, z_geo, y_geo, z_hold, x_hold, params, ell_common_grid, ell_resid_grid):
    # Approximate shared-latent likelihood:
    # cov(D,D)=Kc+Kr+noise, cov(F,F)=lambda^2 Kc+Kz+noise, cov(D,F)=lambda Kc.
    yD = y_direct - np.mean(y_direct)
    yF = y_geo - np.mean(y_geo)
    y = np.r_[yD, yF]
    nD = len(yD)
    nF = len(yF)
    nll = np.empty((len(ell_common_grid), len(ell_resid_grid)))
    for i, ec in enumerate(ell_common_grid):
        for j, er in enumerate(ell_resid_grid):
            cDD = exp_cov(z_direct, z_direct, ec, params["sigma_common"] ** 2)
            rDD = exp_cov(z_direct, z_direct, er, params["sigma_residual"] ** 2)
            cFF = exp_cov(z_geo, z_geo, ec, params["sigma_common"] ** 2)
            zFF = exp_cov(z_geo, z_geo, params["ell_geo_noise"], params["sigma_geo_noise"] ** 2)
            cDF = exp_cov(z_direct, z_geo, ec, params["sigma_common"] ** 2)
            top = np.c_[cDD + rDD + params["sigma_direct_obs"] ** 2 * np.eye(nD), params["lambda_geo"] * cDF]
            bot = np.c_[params["lambda_geo"] * cDF.T, params["lambda_geo"] ** 2 * cFF + zFF + params["sigma_geo_obs"] ** 2 * np.eye(nF)]
            cov = np.r_[top, bot]
            nll[i, j] = profile_nll(y, cov)
    w = np.exp(-0.5 * (nll - nll.min()))
    w = w / w.sum()
    ii, jj = np.unravel_index(np.argmin(nll), nll.shape)
    ec_hat = float(ell_common_grid[ii])
    er_hat = float(ell_resid_grid[jj])
    theta_grid = effective_theta(ell_common_grid[:, None], ell_resid_grid[None, :], params)
    theta_hat = float(theta_grid[ii, jj])
    theta_flat = theta_grid.ravel()
    w_flat = w.ravel()
    order = np.argsort(theta_flat)
    cdf = np.cumsum(w_flat[order])
    q05 = float(theta_flat[order][np.searchsorted(cdf, 0.05)])
    q50 = float(theta_flat[order][np.searchsorted(cdf, 0.50)])
    q95 = float(theta_flat[order][np.searchsorted(cdf, 0.95)])

    # Prediction uses conditional GP at fused MLE parameters.
    cOO = np.block(
        [
            [
                exp_cov(z_direct, z_direct, ec_hat, params["sigma_common"] ** 2)
                + exp_cov(z_direct, z_direct, er_hat, params["sigma_residual"] ** 2)
                + params["sigma_direct_obs"] ** 2 * np.eye(nD),
                params["lambda_geo"] * exp_cov(z_direct, z_geo, ec_hat, params["sigma_common"] ** 2),
            ],
            [
                params["lambda_geo"] * exp_cov(z_geo, z_direct, ec_hat, params["sigma_common"] ** 2),
                params["lambda_geo"] ** 2 * exp_cov(z_geo, z_geo, ec_hat, params["sigma_common"] ** 2)
                + exp_cov(z_geo, z_geo, params["ell_geo_noise"], params["sigma_geo_noise"] ** 2)
                + params["sigma_geo_obs"] ** 2 * np.eye(nF),
            ],
        ]
    )
    cPO = np.c_[
        exp_cov(z_hold, z_direct, ec_hat, params["sigma_common"] ** 2)
        + exp_cov(z_hold, z_direct, er_hat, params["sigma_residual"] ** 2),
        params["lambda_geo"] * exp_cov(z_hold, z_geo, ec_hat, params["sigma_common"] ** 2),
    ]
    pred = cPO @ np.linalg.solve(cOO, y)
    rmse = float(np.sqrt(np.mean((x_hold - pred) ** 2)))
    return {
        "ell_common": ec_hat,
        "ell_residual": er_hat,
        "theta_hat": theta_hat,
        "theta_q05": q05,
        "theta_q50": q50,
        "theta_q95": q95,
        "weights": w,
        "nll": nll,
        "theta_grid": theta_grid,
        "rmse": rmse,
        "pred_hold": pred,
    }


def effective_theta(ell_common, ell_resid, params):
    vc = params["sigma_common"] ** 2
    vr = params["sigma_residual"] ** 2
    # Integral scale for exponential covariance in 1D is 2*ell; weighted by variance shares.
    return 2.0 * (vc * ell_common + vr * ell_resid) / (vc + vr)


def weighted_quantiles(values, weights, probs=(0.05, 0.50, 0.95)):
    values = np.asarray(values).ravel()
    weights = np.asarray(weights).ravel()
    order = np.argsort(values)
    cdf = np.cumsum(weights[order])
    out = []
    for p in probs:
        out.append(float(values[order][np.searchsorted(cdf, p)]))
    return out


def run_one(seed, cross_level, n_direct):
    rng = np.random.default_rng(seed)
    params = {
        "ell_common": 2.4,
        "ell_residual": 0.65,
        "ell_geo_noise": 1.1,
        "sigma_common": 1.0,
        "sigma_residual": 0.55,
        "sigma_geo_noise": 0.35,
        "sigma_direct_obs": 0.18,
        "sigma_geo_obs": 0.16,
        "lambda_geo": cross_level,
        "support_width": 5,
    }
    z = np.linspace(0, 24, 97)
    _, x, zgeo = sample_fields(rng, z, params)
    direct_idx = np.linspace(4, len(z) - 8, n_direct, dtype=int)
    hold_idx = np.setdiff1d(np.arange(0, len(z), 6), direct_idx)
    geo_idx = np.arange(2, len(z) - 2, 2)
    zD, yD = z[direct_idx], x[direct_idx] + rng.normal(0, params["sigma_direct_obs"], n_direct)
    zF, yF = z[geo_idx], zgeo[geo_idx] + rng.normal(0, params["sigma_geo_obs"], len(geo_idx))
    zH, xH = z[hold_idx], x[hold_idx]
    length_grid = np.linspace(0.25, 8.0, 36)
    resid_grid = np.linspace(0.20, 2.40, 24)
    common_grid = np.linspace(0.45, 6.00, 30)

    theta_true = float(effective_theta(params["ell_common"], params["ell_residual"], params))

    emp_ell = empirical_corr_length(zD, yD)
    emp_pred, _ = gp_condition(zD, yD - np.mean(yD), zH, emp_ell, max(np.var(yD), 1e-4), params["sigma_direct_obs"] ** 2)
    emp_rmse = float(np.sqrt(np.mean((xH - emp_pred) ** 2)))

    direct_ell, direct_w = estimate_direct_ml(zD, yD, length_grid, params["sigma_direct_obs"])
    direct_mean, direct_var = gp_condition(zD, yD - np.mean(yD), zH, direct_ell, max(np.var(yD), 1e-4), params["sigma_direct_obs"] ** 2)
    direct_rmse = float(np.sqrt(np.mean((xH - direct_mean) ** 2)))
    d_q05, d_q50, d_q95 = weighted_quantiles(2.0 * length_grid, direct_w)

    proxy_ell, proxy_w = estimate_proxy_length(zF, yF, length_grid, params["sigma_geo_obs"])
    p_q05, p_q50, p_q95 = weighted_quantiles(2.0 * length_grid, proxy_w)
    geo_at_direct = interp_at(zF, yF, zD)
    a_proxy, b_proxy = np.polyfit(geo_at_direct, yD, 1)
    proxy_pred = a_proxy * interp_at(zF, yF, zH) + b_proxy
    proxy_rmse = float(np.sqrt(np.mean((xH - proxy_pred) ** 2)))

    reg_ell, reg_rmse, reg_pred, reg_a, reg_b = regression_assisted(zD, yD, zF, yF, zH, xH, length_grid, params["sigma_direct_obs"])

    fusion = fusion_grid(zD, yD, zF, yF, zH, xH, params, common_grid, resid_grid)

    rows = [
        {
            "method": "Direct empirical variogram",
            "theta_median_m": 2 * emp_ell,
            "theta_q05_m": np.nan,
            "theta_q95_m": np.nan,
            "abs_log_error": abs(math.log((2 * emp_ell) / theta_true)),
            "holdout_rmse": emp_rmse,
            "interval_width_m": np.nan,
        },
        {
            "method": "Direct-only ML GP",
            "theta_median_m": d_q50,
            "theta_q05_m": d_q05,
            "theta_q95_m": d_q95,
            "abs_log_error": abs(math.log(d_q50 / theta_true)),
            "holdout_rmse": direct_rmse,
            "interval_width_m": d_q95 - d_q05,
        },
        {
            "method": "Geophysics-proxy-only",
            "theta_median_m": p_q50,
            "theta_q05_m": p_q05,
            "theta_q95_m": p_q95,
            "abs_log_error": abs(math.log(p_q50 / theta_true)),
            "holdout_rmse": proxy_rmse,
            "interval_width_m": p_q95 - p_q05,
        },
        {
            "method": "Regression-assisted residual GP",
            "theta_median_m": 2 * reg_ell,
            "theta_q05_m": np.nan,
            "theta_q95_m": np.nan,
            "abs_log_error": abs(math.log((2 * reg_ell) / theta_true)),
            "holdout_rmse": reg_rmse,
            "interval_width_m": np.nan,
        },
        {
            "method": "Shared latent-field fusion",
            "theta_median_m": fusion["theta_q50"],
            "theta_q05_m": fusion["theta_q05"],
            "theta_q95_m": fusion["theta_q95"],
            "abs_log_error": abs(math.log(fusion["theta_q50"] / theta_true)),
            "holdout_rmse": fusion["rmse"],
            "interval_width_m": fusion["theta_q95"] - fusion["theta_q05"],
        },
    ]
    return {
        "params": params,
        "z": z,
        "x": x,
        "zgeo": zgeo,
        "zD": zD,
        "yD": yD,
        "zF": zF,
        "yF": yF,
        "zH": zH,
        "xH": xH,
        "theta_true": theta_true,
        "rows": rows,
        "fusion": fusion,
        "direct_posterior": (2.0 * length_grid, direct_w),
        "proxy_posterior": (2.0 * length_grid, proxy_w),
        "predictions": {
            "direct": direct_mean,
            "proxy": proxy_pred,
            "regression": reg_pred,
            "fusion": fusion["pred_hold"],
        },
    }


def fig_benchmark(case):
    fig, ax = plt.subplots(figsize=(8.2, 4.7), dpi=220)
    ax.plot(case["z"], case["x"], color="#1b1b1b", lw=1.8, label="True geotechnical field X(z)")
    ax.plot(case["z"], case["zgeo"], color="#3b7ea1", lw=1.5, label="Dense geophysical response Z(z)")
    ax.scatter(case["zD"], case["yD"], s=42, marker="o", color="#b23a2e", edgecolor="white", zorder=5, label="Sparse CPT/SPT observations")
    ax.scatter(case["zF"], case["yF"], s=14, color="#3b7ea1", alpha=0.45, label="Geophysical samples")
    ax.set_xlabel("Depth z (m)", fontsize=11)
    ax.set_ylabel("Standardized response", fontsize=11)
    ax.set_title("Synthetic infrastructure-site benchmark", fontsize=13, weight="bold")
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax.grid(True, lw=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIG / "Figure 1 synthetic benchmark.png")
    plt.close(fig)


def fig_methods(case, df):
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=220)
    colors = ["#636363", "#2c6b9a", "#b76e2a", "#6a994e", "#7c3f8c"]
    x = np.arange(len(df))
    ax.bar(x, df["abs_log_error"], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in df["method"]], fontsize=8.5)
    ax.set_ylabel("Absolute log error in effective scale", fontsize=10.5)
    ax.set_title("Recovery of the geotechnical correlation length", fontsize=13, weight="bold")
    ax.grid(axis="y", lw=0.3, alpha=0.35)
    for i, val in enumerate(df["abs_log_error"]):
        ax.text(i, val + 0.015, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "Figure 2 method comparison.png")
    plt.close(fig)


def fig_posterior(case):
    direct_theta, direct_w = case["direct_posterior"]
    proxy_theta, proxy_w = case["proxy_posterior"]
    fusion_theta = case["fusion"]["theta_grid"].ravel()
    fusion_w = case["fusion"]["weights"].ravel()
    order = np.argsort(fusion_theta)
    fig, ax = plt.subplots(figsize=(8.2, 4.5), dpi=220)
    ax.plot(direct_theta, direct_w / direct_w.max(), color="#2c6b9a", lw=1.8, label="Direct-only ML GP")
    ax.plot(proxy_theta, proxy_w / proxy_w.max(), color="#b76e2a", lw=1.8, label="Geophysics proxy")
    # Smoothed histogram for fusion
    bins = np.linspace(min(fusion_theta), max(fusion_theta), 70)
    hist, edges = np.histogram(fusion_theta, bins=bins, weights=fusion_w, density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, hist / hist.max(), color="#7c3f8c", lw=2.2, label="Shared latent-field fusion")
    ax.axvline(case["theta_true"], color="#1b1b1b", ls="--", lw=1.5, label="True scale")
    ax.set_xlabel("Effective geotechnical correlation length ΘX,z (m)", fontsize=10.5)
    ax.set_ylabel("Relative posterior density", fontsize=10.5)
    ax.set_title("Posterior concentration after data fusion", fontsize=13, weight="bold")
    ax.legend(fontsize=8.8, frameon=True)
    ax.grid(True, lw=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIG / "Figure 3 posterior scale.png")
    plt.close(fig)


def fig_decision(case, df):
    # Infrastructure proxy: number of direct soundings required to get interval width below a design threshold.
    scenarios = ["Sparse direct only", "Fusion with geophysics"]
    widths = [
        float(df.loc[df["method"] == "Direct-only ML GP", "interval_width_m"].iloc[0]),
        float(df.loc[df["method"] == "Shared latent-field fusion", "interval_width_m"].iloc[0]),
    ]
    target_width = 3.0
    required = [math.ceil(w / target_width * len(case["zD"])) for w in widths]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=220)
    bars = ax.bar(scenarios, required, color=["#2c6b9a", "#6a994e"])
    ax.set_ylabel("Equivalent direct soundings for target uncertainty", fontsize=10.5)
    ax.set_title("Infrastructure exploration value of geophysical fusion", fontsize=12.5, weight="bold")
    ax.grid(axis="y", lw=0.3, alpha=0.35)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f"{int(b.get_height())}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "Figure 4 infrastructure decision value.png")
    plt.close(fig)
    return required


def infrastructure_reliability(theta_m):
    # Simple downstream civil-infrastructure consequence metric.
    # The resistance margin is averaged over a foundation/slope mechanism length L.
    # Longer correlation length reduces spatial averaging and increases the variance
    # of the mechanism-average resistance. This is not a code check; it is a
    # transparent propagation of length-scale uncertainty into a reliability proxy.
    mechanism_length = 15.0
    mean_safety_margin = 1.35
    point_cov = 0.18
    model_sigma = 0.04
    gamma2 = min(1.0, max(0.0, 2.0 * theta_m / mechanism_length))
    sigma_margin = math.sqrt((mean_safety_margin * point_cov * math.sqrt(gamma2)) ** 2 + model_sigma**2)
    beta = (mean_safety_margin - 1.0) / sigma_margin
    pf = 0.5 * math.erfc(beta / math.sqrt(2.0))
    return beta, pf


def reliability_table(primary_df):
    rows = []
    for label in ["Direct-only ML GP", "Geophysics-proxy-only", "Shared latent-field fusion"]:
        row = primary_df.loc[primary_df["method"] == label].iloc[0]
        vals = {
            "5% Theta (m)": row["theta_q05_m"],
            "Median Theta (m)": row["theta_median_m"],
            "95% Theta (m)": row["theta_q95_m"],
        }
        beta_med, pf_med = infrastructure_reliability(float(row["theta_median_m"]))
        beta_low, pf_low = infrastructure_reliability(float(row["theta_q05_m"]))
        beta_high, pf_high = infrastructure_reliability(float(row["theta_q95_m"]))
        rows.append(
            {
                "method": label,
                "theta_q05_m": vals["5% Theta (m)"],
                "theta_median_m": vals["Median Theta (m)"],
                "theta_q95_m": vals["95% Theta (m)"],
                "beta_median": beta_med,
                "pf_median": pf_med,
                "pf_q05_theta": pf_low,
                "pf_q95_theta": pf_high,
            }
        )
    return pd.DataFrame(rows)


def fig_reliability(rel_df):
    labels = rel_df["method"].str.replace(" ", "\n")
    pf = rel_df["pf_median"].values
    low = np.minimum(rel_df["pf_q05_theta"].values, rel_df["pf_q95_theta"].values)
    high = np.maximum(rel_df["pf_q05_theta"].values, rel_df["pf_q95_theta"].values)
    yerr = np.vstack([pf - low, high - pf])
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=220)
    x = np.arange(len(labels))
    ax.bar(x, pf, color=["#2c6b9a", "#b76e2a", "#6a994e"], alpha=0.9)
    ax.errorbar(x, pf, yerr=yerr, fmt="none", ecolor="#1b1b1b", capsize=5, lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Probability of margin exceedance Pf", fontsize=10.5)
    ax.set_title("Downstream reliability effect of correlation-length uncertainty", fontsize=12.5, weight="bold")
    ax.grid(axis="y", lw=0.3, alpha=0.35)
    for i, val in enumerate(pf):
        ax.text(i, val + 0.003, f"{100*val:.1f}%", ha="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "Figure 5 downstream reliability.png")
    plt.close(fig)


def save_table_image(df, filename, title):
    fig_h = 1.1 + 0.42 * (len(df) + 1)
    fig, ax = plt.subplots(figsize=(10.6, fig_h), dpi=220)
    ax.axis("off")
    ax.set_title(title, fontsize=12, weight="bold", pad=10, color="black")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="left", colLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.2)
    tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.set_linewidth(0.4)
        cell.set_facecolor("white")
        if r == 0:
            cell.set_text_props(weight="bold", color="black")
    fig.tight_layout()
    fig.savefig(TAB / filename)
    plt.close(fig)


def main():
    primary = run_one(seed=12264, cross_level=0.55, n_direct=5)
    rows = []
    seeds = range(12200, 12208)
    levels = {"weak": 0.25, "moderate": 0.55, "strong": 0.85}
    for label, lam in levels.items():
        for nd in (5, 6, 8):
            for seed in seeds:
                case = run_one(seed=seed + int(lam * 100) + nd, cross_level=lam, n_direct=nd)
                for r in case["rows"]:
                    rr = dict(r)
                    rr["scenario"] = label
                    rr["n_direct"] = nd
                    rr["seed"] = seed
                    rr["theta_true_m"] = case["theta_true"]
                    rows.append(rr)
    all_df = pd.DataFrame(rows)
    all_df.to_csv(DATA / "benchmark_all_replicates.csv", index=False)

    primary_df = pd.DataFrame(primary["rows"])
    primary_df.to_csv(DATA / "primary_case_method_results.csv", index=False)
    primary_series = pd.DataFrame(
        {
            "depth_m": primary["z"],
            "true_geotechnical_field": primary["x"],
            "dense_geophysical_response": primary["zgeo"],
        }
    )
    primary_series.to_csv(DATA / "primary_case_profiles.csv", index=False)

    summary = (
        all_df.groupby(["scenario", "n_direct", "method"], as_index=False)
        .agg(
            median_abs_log_error=("abs_log_error", "median"),
            median_holdout_rmse=("holdout_rmse", "median"),
            mean_interval_width_m=("interval_width_m", "mean"),
        )
        .sort_values(["scenario", "n_direct", "median_abs_log_error"])
    )
    summary.to_csv(DATA / "benchmark_summary_by_scenario.csv", index=False)

    fig_benchmark(primary)
    fig_methods(primary, primary_df)
    fig_posterior(primary)
    required_soundings = fig_decision(primary, primary_df)
    rel_df = reliability_table(primary_df)
    rel_df.to_csv(DATA / "infrastructure_reliability_proxy.csv", index=False)
    fig_reliability(rel_df)

    table1 = pd.DataFrame(
        [
            ["z", "Depth coordinate", "m", "Infrastructure-site vertical profile"],
            ["X(z)", "Geotechnical design property", "standardized", "Target field inferred from CPT/SPT"],
            ["Z(z)", "Geophysical response", "standardized", "Dense indirect observation"],
            ["E(z)", "Common latent geological component", "standardized", "Shared by geotechnical and geophysical fields"],
            ["RX(z)", "Geotechnical residual", "standardized", "Not directly visible to geophysics"],
            ["ΘX,z", "Effective geotechnical correlation length", "m", "Quantity used in reliability and exploration planning"],
        ],
        columns=["Symbol", "Definition", "Unit", "Engineering role"],
    )
    table1.to_csv(DATA / "table_1_variables.csv", index=False)
    save_table_image(table1, "Table 1 variables.png", "Table 1. Variables used in the shared latent-field benchmark.")

    table2 = pd.DataFrame(
        [
            ["Common latent scale", "2.40 m"],
            ["Geotechnical residual scale", "0.65 m"],
            ["Geophysical residual scale", "1.10 m"],
            ["Common-field standard deviation", "1.00"],
            ["Geotechnical residual standard deviation", "0.55"],
            ["Direct CPT/SPT observation noise", "0.18"],
            ["Dense geophysical observation noise", "0.16"],
            ["Primary case direct soundings", "5"],
            ["Primary case geophysical samples", "46"],
        ],
        columns=["Benchmark parameter", "Value"],
    )
    table2.to_csv(DATA / "table_2_benchmark_parameters.csv", index=False)
    save_table_image(table2, "Table 2 benchmark parameters.png", "Table 2. Synthetic benchmark parameters.")

    compact_primary = primary_df.copy()
    for col in ["theta_median_m", "theta_q05_m", "theta_q95_m", "abs_log_error", "holdout_rmse", "interval_width_m"]:
        compact_primary[col] = compact_primary[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    save_table_image(compact_primary, "Table 3 primary results.png", "Table 3. Primary benchmark results by method.")

    compact_summary = summary[
        (summary["scenario"].isin(["weak", "moderate", "strong"])) & (summary["n_direct"] == 6)
    ].copy()
    for col in ["median_abs_log_error", "median_holdout_rmse", "mean_interval_width_m"]:
        compact_summary[col] = compact_summary[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    save_table_image(compact_summary, "Table 4 scenario sensitivity.png", "Table 4. Sensitivity to cross-information strength.")

    compact_rel = rel_df.copy()
    compact_rel = compact_rel[
        ["method", "theta_median_m", "beta_median", "pf_median", "pf_q05_theta", "pf_q95_theta"]
    ]
    compact_rel.columns = [
        "Method",
        "Median Theta (m)",
        "Median beta",
        "Median Pf",
        "Pf at 5% Theta",
        "Pf at 95% Theta",
    ]
    for col in ["Median Theta (m)", "Median beta", "Median Pf", "Pf at 5% Theta", "Pf at 95% Theta"]:
        compact_rel[col] = compact_rel[col].map(lambda v: f"{v:.4f}")
    save_table_image(compact_rel, "Table 5 reliability propagation.png", "Table 5. Downstream reliability propagation.")

    best_fusion = primary_df.loc[primary_df["method"] == "Shared latent-field fusion"].iloc[0]
    direct = primary_df.loc[primary_df["method"] == "Direct-only ML GP"].iloc[0]
    interval_reduction = 1.0 - best_fusion["interval_width_m"] / direct["interval_width_m"]
    rmse_reduction = 1.0 - best_fusion["holdout_rmse"] / direct["holdout_rmse"]
    out_summary = {
        "seed_primary": 12264,
        "true_effective_theta_m": primary["theta_true"],
        "selected_target_journal": "Engineering Geology",
        "primary_fusion_theta_median_m": float(best_fusion["theta_median_m"]),
        "primary_direct_theta_median_m": float(direct["theta_median_m"]),
        "interval_width_reduction_vs_direct": float(interval_reduction),
        "holdout_rmse_reduction_vs_direct": float(rmse_reduction),
        "equivalent_direct_soundings_sparse_direct": int(required_soundings[0]),
        "equivalent_direct_soundings_fusion": int(required_soundings[1]),
        "primary_fusion_reliability_beta": float(rel_df.loc[rel_df["method"] == "Shared latent-field fusion", "beta_median"].iloc[0]),
        "primary_fusion_reliability_pf": float(rel_df.loc[rel_df["method"] == "Shared latent-field fusion", "pf_median"].iloc[0]),
        "replicates": int(len(seeds) * len(levels) * 3),
        "methods": sorted(all_df["method"].unique().tolist()),
        "outputs": {
            "figures": [str(p.relative_to(OUT)) for p in sorted(FIG.glob("*.png"))],
            "table_images": [str(p.relative_to(OUT)) for p in sorted(TAB.glob("*.png"))],
            "data": [str(p.relative_to(OUT)) for p in sorted(DATA.glob("*.csv"))],
        },
    }
    (OUT / "benchmark_summary.json").write_text(json.dumps(out_summary, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Shared latent-field benchmark for article 122\n\n"
        "This folder contains a fully reproducible synthetic benchmark for a manuscript "
        "on Bayesian estimation of geotechnical correlation lengths by geotechnical-geophysical fusion.\n\n"
        "Run:\n\n"
        "```powershell\n"
        "python benchmark_latent_fusion.py\n"
        "```\n\n"
        "The script generates random fields with known correlation lengths, sparse direct CPT/SPT-like "
        "observations, dense geophysical observations and four baseline comparators plus the proposed "
        "shared latent-field fusion model. Figures and table images are generated from the CSV outputs.\n\n"
        f"Primary true effective correlation length: {primary['theta_true']:.3f} m.\n"
        f"Fusion median estimate: {float(best_fusion['theta_median_m']):.3f} m.\n"
        f"Direct-only median estimate: {float(direct['theta_median_m']):.3f} m.\n"
        f"90% interval width reduction versus direct-only: {100*interval_reduction:.1f}%.\n"
        f"Holdout RMSE reduction versus direct-only: {100*rmse_reduction:.1f}%.\n",
        encoding="utf-8",
    )
    print(json.dumps(out_summary, indent=2))


if __name__ == "__main__":
    main()
