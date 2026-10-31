"""Paper figures v4 (static, print-safe). Text = Computer Modern 10 pt (same as the body), figures saved at their physical size and included at scale 1:1.
Data: suite/*.pkl (policy_suite.py), r3/*.txt (fleet_eval), logs of the label-scarcity experiment. Nothing is hard-coded except numbers copied from named logs."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import WORK, FIGS, CMAPSS_ROOT, PRON_ROOT
import re, json, pickle, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator, FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
import matplotlib.patheffects as pe
HALO = [pe.withStroke(linewidth=3.2, foreground='white')]
plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm", "axes.unicode_minus": False, "axes.formatter.use_mathtext": True,
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "figure.constrained_layout.use": True,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.labelcolor": "#0b0b0b", "grid.color": "#e4e3df", "grid.linewidth": 0.5, "legend.frameon": False})
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#8a8984", "#0b0b0b"
SRC, OUT = WORK + "/", FIGS + "/"
def save(fig, name): fig.savefig(OUT + name + ".png", dpi=300); fig.savefig(OUT + name + ".pdf"); plt.close(fig)
def clean_log(ax, xt=None, yt=None):
    if xt is not None: ax.xaxis.set_major_locator(FixedLocator(xt)); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    if yt is not None: ax.yaxis.set_major_locator(FixedLocator(yt)); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
def panel(ax, text): ax.set_title(text, loc="left", pad=4)
FL = ["FD001", "FD002", "FD003", "FD004"]; CU = [2, 5, 10, 20]
PRED = ["ridge", "hgb", "rf", "mlp", "gru", "cnn", "paris"]
PLAB = {"ridge": "Ridge", "hgb": "Hist-GB", "rf": "Random forest", "mlp": "MLP", "gru": "GRU", "cnn": "1D-CNN", "paris": "Paris-law hybrid", "ssm": "Wear-law state space"}
SU = {(n, f): pickle.load(open(f"{SRC}suite/{n}_{f}.pkl", "rb")) for n in PRED for f in FL}
def fmean(n, pol, cu):                                   # mean over engines per fleet, then mean over the four fleets
    return np.mean([np.nanmean(SU[(n, f)]["per"][pol][cu]) for f in FL])
POLICY_STYLE = {"point": ("point RUL", ORANGE, "--", "s"), "a50": ("safety 0.5", YELLOW, ":", "P"), "a70": ("safety 0.7", MAGENTA, "-.", "X"), "f90": ("fixed 90%", AQUA, "-.", "^"),
                "gauss": ("Gauss quantile", VIOLET, "--", "D"), "static": ("static $\\tau^*$", GREY, ":", "v"), "cum": ("$\\tau^*$-cumulative (ours)", BLUE, "-", "o"), "oracle": ("population quantile", INK, ":", "*")}

# ====================================================================================================================
# Fig 2: distribution of the ratio rho = r/p at the alarm, with the tau*-quantile for each cost ratio (ridge plot)
# ====================================================================================================================
fig, ax = plt.subplots(figsize=(7.53, 2.9)); grid = np.linspace(0.12, 3.2, 600); cmap = LinearSegmentedColormap.from_list("b", ["#b9d4f5", "#0d3b7a"]); cols = [cmap(v) for v in np.linspace(0, 1, 4)]
for i, f in enumerate(FL):
    rho = SU[("mlp", f)]["ratios"]; kde = gaussian_kde(np.log(rho), bw_method=0.25); dens = kde(np.log(grid)) / grid; dens = dens / dens.max() * 0.82; y0 = (len(FL) - 1 - i)
    ax.fill_between(grid, y0, y0 + dens, color="#dfe9f8", lw=0, zorder=2); ax.plot(grid, y0 + dens, color="#52514e", lw=0.9, zorder=3)
    q20 = np.quantile(rho, 1 / 21); m = grid <= q20; ax.fill_between(grid[m], y0, y0 + dens[m], color=cols[3], alpha=0.55, lw=0, zorder=3)      # left tail for c_u/c_o = 20
    for cu, c in zip(CU, cols):
        q = np.quantile(rho, 1 / (1 + cu)); h = np.interp(q, grid, dens); ax.plot([q, q], [y0, y0 + h + 0.03], color=c, lw=2.2, solid_capstyle="butt", zorder=4)
    ax.text(3.18, y0 + 0.30, f, ha="right", va="center", color="#52514e")
    ax.plot(np.median(rho), y0, marker="|", ms=7, color=INK, zorder=5)
ax.axvline(1.0, color=ORANGE, ls="--", lw=1.1, zorder=1); ax.text(1.04, len(FL) - 0.34, "point RUL ($q=1$)", color=ORANGE, ha="left", va="bottom")
ax.set_xlim(0.1, 3.2); ax.set_ylim(-0.05, len(FL) + 0.15); ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlabel("ratio $\\rho=r/p$ of true to predicted RUL at the alarm"); ax.grid(True, axis="x")
ax.text(0.14, len(FL) - 0.02, "numbers: $c_u/c_o$ of the\\\\quantile $q^*$ (level $\\tau^*$)", ha="left", va="top", fontsize=10, color="#52514e") if False else None
ax.text(0.14, len(FL) + 0.02, "coloured ticks: $q^*$ for $c_u/c_o=2,5,10,20$ (light to dark);  | = median", ha="left", va="bottom", color="#52514e", fontsize=10)
save(fig, "fig_ratio_ridge")

# ====================================================================================================================
# Fig 3: reliability (achieved minus nominal coverage) of three calibration methods, per fleet
# ====================================================================================================================
fig, axs = plt.subplots(1, 4, figsize=(7.53, 2.55), sharex=True, sharey=True); nom = 1 - SU[("mlp", "FD001")]["TAUS"]
for ax, f in zip(axs, FL):
    d = SU[("mlp", f)]["sweeps"]; ax.axhspan(-0.02, 0.02, color="#eceae5", lw=0, zorder=0); ax.axhline(0, color="#52514e", lw=0.8, zorder=1)
    for key, col, ls, mk, lab in (("static", GREY, ":", "v", "static"), ("gauss", VIOLET, "--", "D", "Gauss quantile"), ("cum", BLUE, "-", "o", "conformal, cumulative")):
        ax.plot(nom, d[key]["cov"] - nom, color=col, ls=ls, lw=1.7 if key == "cum" else 1.3, marker=mk, ms=3.6, mec="white", mew=0.4, zorder=3 if key == "cum" else 2, label=lab)
    panel(ax, f); ax.grid(True); ax.set_ylim(-0.075, 0.045); ax.set_xlim(0.37, 0.98)
for ax in axs: ax.set_xlabel("nominal $1-\\tau$")
axs[0].set_ylabel("achieved $-$ nominal")
fig.legend(handles=[Line2D([], [], color=BLUE, lw=1.7, marker="o", ms=3.6, label="conformal (ours)"), Line2D([], [], color=VIOLET, lw=1.3, ls="--", marker="D", ms=3.6, label="Gauss"), Line2D([], [], color=GREY, lw=1.3, ls=":", marker="v", ms=3.6, label="static")],
           loc="outside upper center", ncol=3, handlelength=1.8, columnspacing=1.0)
save(fig, "fig_reliability")

# ====================================================================================================================
# Fig 4: summary curves. (a) all policies (MLP, mean of four fleets)  (b) the proposed policy under every predictor
# ====================================================================================================================
fig, axs = plt.subplots(1, 2, figsize=(7.53, 3.1))
ax = axs[0]
for pol in ("a50", "a70", "point", "f90", "gauss", "static", "oracle", "cum"):
    lab, col, ls, mk = POLICY_STYLE[pol]; y = [fmean("mlp", pol, c) for c in CU]
    if pol == "cum":
        lo = [min(np.nanmean(SU[("mlp", f)]["per"]["cum"][c]) for f in FL) for c in CU]; hi = [max(np.nanmean(SU[("mlp", f)]["per"]["cum"][c]) for f in FL) for c in CU]; ax.fill_between(CU, lo, hi, color=BLUE, alpha=0.18, lw=0)
    ax.plot(CU, y, color=col, ls=ls, lw=2.0 if pol == "cum" else 1.3, marker=mk, ms=5 if pol == "cum" else 4, mec="white", mew=0.5, label=lab, zorder=5 if pol == "cum" else 3)
ax.set_xscale("log"); ax.set_yscale("log"); clean_log(ax, CU, [0.25, 0.5, 1, 2]); ax.set_ylim(0.2, 14.0); ax.grid(True, which="major"); ax.set_xlabel("$c_u/c_o$"); ax.set_ylabel("cost per alarm / $P_0$"); panel(ax, "(a) all policies, MLP predictor")
ax.legend(loc="upper left", ncol=2, handlelength=2.0, labelspacing=0.2, columnspacing=1.0, borderaxespad=0.2)
ax = axs[1]; pc = {"ridge": GREY, "hgb": YELLOW, "rf": AQUA, "mlp": BLUE, "gru": VIOLET, "cnn": MAGENTA, "paris": ORANGE}; pm = {"ridge": "v", "hgb": "P", "rf": "^", "mlp": "o", "gru": "D", "cnn": "X", "paris": "s"}
for n in PRED:
    ax.plot(CU, [fmean(n, "cum", c) for c in CU], color=pc[n], lw=2.0 if n == "mlp" else 1.4, marker=pm[n], ms=4.5, mec="white", mew=0.5, label=PLAB[n], zorder=4 if n == "mlp" else 3)
ax.set_xscale("log"); ax.set_yscale("log"); clean_log(ax, CU, [0.25, 0.3, 0.4, 0.5, 0.7]); ax.set_ylim(0.22, 1.25); ax.grid(True, which="major"); ax.set_xlabel("$c_u/c_o$"); ax.set_ylabel("cost per alarm / $P_0$"); panel(ax, "(b) $\\tau^*$-cumulative policy, all predictors")
ax.legend(loc="upper left", ncol=2, handlelength=2.0, labelspacing=0.2, columnspacing=1.0, borderaxespad=0.2)
save(fig, "fig_summary")

# ====================================================================================================================
# Fig 5: heat map predictor x policy at two cost ratios
# ====================================================================================================================
COLS = ["point", "a50", "a70", "f90", "gauss", "static", "cum", "oracle"]; CLAB = ["point\nRUL", "safety\n0.5", "safety\n0.7", "fixed\n90%", "Gauss\nquantile", "static\n$\\tau^*$", "$\\tau^*$-cum.\n(ours)", "popul.\nquantile"]
cm = LinearSegmentedColormap.from_list("seq", ["#f4f8fe", "#a9c9f2", "#2a78d6", "#0b2d63"]); norm = LogNorm(vmin=0.22, vmax=5.0)
fig, axs = plt.subplots(2, 1, figsize=(7.53, 6.3), sharex=True)
for ax, cu, tt in zip(axs, (2, 20), ("(a) $c_u/c_o=2$", "(b) $c_u/c_o=20$")):
    M = np.array([[fmean(n, p, cu) for p in COLS] for n in PRED]); ax.imshow(M, cmap=cm, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white" if norm(M[i, j]) > 0.52 else INK, fontsize=10)
        jb = int(np.nanargmin(M[i, :-1])); ax.add_patch(Rectangle((jb - 0.5, i - 0.5), 1, 1, fill=False, ec=ORANGE, lw=1.8))   # best deployable policy (the population quantile uses future units)
    ax.set_yticks(range(len(PRED))); ax.set_yticklabels([PLAB[n] for n in PRED]); ax.set_xticks(range(len(COLS))); ax.set_xticklabels(CLAB, fontsize=10); ax.tick_params(length=0)
    ax.set_title(tt, loc="left", pad=4)
    for s in ax.spines.values(): s.set_visible(False)
    ax.add_patch(Rectangle((COLS.index("cum") - 0.5, -0.5), 1, len(PRED), fill=False, ec=BLUE, lw=1.2, ls="--"))
axs[0].tick_params(labeltop=False, labelbottom=False)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), ax=axs, orientation="horizontal", fraction=0.05, pad=0.03, aspect=45); cb.set_label("cost per alarm / $P_0$ (mean of four fleets)"); cb.set_ticks([0.25, 0.5, 1, 2, 4]); cb.set_ticklabels(["0.25", "0.5", "1", "2", "4"]); cb.ax.xaxis.set_minor_formatter(NullFormatter()); cb.ax.tick_params(which="minor", length=0)
save(fig, "fig_heatmap")

# ====================================================================================================================
# Fig 6: risk-waste frontier with iso-cost lines (pooled over the four fleets)
# ====================================================================================================================
def pooled_sweep(n, key, what): return np.mean([SU[(n, f)]["sweeps"][key][what] for f in FL], axis=0)
def pooled_ovw(n, pol, cu): return np.mean([SU[(n, f)]["ovw"][cu][pol] for f in FL], axis=0)
fig, axs = plt.subplots(1, 2, figsize=(7.53, 3.15), sharey=False)
for ax, n, tt in zip(axs, ("mlp", "ridge"), ("(a) MLP", "(b) Ridge, weakest predictor")):
    for key, col, ls in (("alpha", ORANGE, "--"), ("gauss", VIOLET, "--"), ("static", GREY, ":"), ("cum", BLUE, "-")):
        ax.plot(pooled_sweep(n, key, "waste"), pooled_sweep(n, key, "over"), color=col, ls=ls, lw=2.4 if key == "cum" else 1.2, zorder=4 if key == "cum" else 2, alpha=0.9 if key == "cum" else 0.8)
    # fixed multipliers (their operating point does not move with the cost ratio)
    for pol, lab in (("point", "1"), ("a85", ".85"), ("a70", ".7"), ("a50", ".5")):
        o, w = pooled_ovw(n, pol, 2); ax.plot(w, o, marker="s", ms=6, color=ORANGE, mec="white", mew=0.6, ls="none", zorder=6); ax.text(w - 0.012, o + 0.006, "$q=$" + lab if pol == "point" else lab, color="#b0481f", ha="right", va="bottom", fontsize=10, path_effects=HALO, zorder=8)
    for cu, c in zip(CU, ["#7fb0ee", "#3f86d8", "#1c5aa8", "#0b2d63"]):
        o, w = pooled_ovw(n, "cum", cu); ax.plot(w, o, marker="o", ms=7.5, color=c, mec="white", mew=0.8, zorder=7); og, wg = pooled_ovw(n, "gauss", cu); ax.plot(wg, og, marker="D", ms=4.5, mfc="none", mec=VIOLET, mew=1.2, ls="none", zorder=6)
        xs = np.linspace(0, 1.1, 50); ax.plot(xs, np.maximum((o * cu + w - xs) / cu, 0), color=c, lw=0.8, alpha=0.8, zorder=1)                # iso-cost line through the operating point
    ax.set_xlim(0, 0.52 if n == "mlp" else 0.78); ax.set_ylim(0, 0.17 if n == "mlp" else 0.21); ax.grid(True); ax.set_xlabel("expected waste $E(r-d)^+/P_0$"); panel(ax, tt)
axs[0].set_ylabel("expected overrun $E(d-r)^+/P_0$")
fig.legend(handles=[Line2D([], [], color=BLUE, lw=2.4, marker="o", ms=7, label="conformal, $\\tau^*$ points (ours)"), Line2D([], [], color=VIOLET, lw=1.2, ls="--", marker="D", ms=4.5, mfc="none", label="Gauss quantile"), Line2D([], [], color=ORANGE, lw=1.2, ls="--", marker="s", ms=6, label="safety factor $q$ (squares)"),
                    Line2D([], [], color=GREY, lw=1.2, ls=":", label="static"), Line2D([], [], color="#52514e", lw=0.8, label="iso-cost line")],
           loc="outside upper center", ncol=2, handlelength=2.4, columnspacing=1.0)
save(fig, "fig_frontier")

# ====================================================================================================================
# Fig 7: per-engine cost distributions at c_u/c_o = 20 (FD002 and FD004, MLP)
# ====================================================================================================================
fig, axs = plt.subplots(1, 2, figsize=(7.53, 3.0), sharey=True); VP = ["point", "a70", "f90", "gauss", "cum"]; rng = np.random.default_rng(0)
for ax, f in zip(axs, ("FD002", "FD004")):
    d = SU[("mlp", f)]["per"]
    for k, pol in enumerate(VP):
        v = d[pol][20]; v = v[~np.isnan(v)]; lab, col, ls, mk = POLICY_STYLE[pol]; lv = np.log10(v); lo_, hi_ = np.quantile(lv, [0.01, 0.99])
        kde = gaussian_kde(lv, bw_method=0.35); yy = np.linspace(lo_ - 0.08, hi_ + 0.08, 200); w = kde(yy); w = w / w.max() * 0.42
        ax.fill_betweenx(10 ** yy, k - w, k + w, color=col, alpha=0.30, lw=0); ax.plot(k + w, 10 ** yy, color=col, lw=0.8); ax.plot(k - w, 10 ** yy, color=col, lw=0.8)
        ax.scatter(k + rng.uniform(-0.12, 0.12, len(v)), v, s=3, color=col, alpha=0.35, lw=0, zorder=3)
        ax.plot([k - 0.3, k + 0.3], [np.median(v)] * 2, color=INK, lw=1.4, zorder=5); ax.plot(k, v.mean(), marker="D", ms=5, color="white", mec=INK, mew=1.0, zorder=6)
    ax.set_yscale("log"); clean_log(ax, None, [0.03, 0.1, 0.3, 1, 3, 10]); ax.set_ylim(0.025, 16); ax.set_xticks(range(len(VP))); ax.set_xticklabels(["point\nRUL", "safety\n0.7", "fixed\n90%", "Gauss\nquantile", "ours"]); ax.grid(True, axis="y"); panel(ax, f"({'ab'[('FD002','FD004').index(f)]}) {f}"); ax.set_xlim(-0.6, len(VP) - 0.4)
axs[0].set_ylabel("per-engine cost / $P_0$,  $c_u/c_o=20$")
fig.legend(handles=[Line2D([], [], color=INK, lw=1.4, label="median"), Line2D([], [], ls="none", marker="D", ms=5, mfc="white", mec=INK, label="mean")], loc="outside upper center", ncol=2, columnspacing=1.5)
save(fig, "fig_violin")

# ====================================================================================================================
# Fig 8: excess cost times n for the four fleets, with the asymptotic constants (smoothed bootstrap of the alarm ratios)
# ====================================================================================================================
STATS8 = {}; rng = np.random.default_rng(5); NS = [5, 10, 20, 40, 80, 160, 320]; fig, axs = plt.subplots(1, 2, figsize=(7.53, 2.55), sharey=False); fcol = {"FD001": BLUE, "FD002": ORANGE, "FD003": AQUA, "FD004": VIOLET}; fmk = {"FD001": "o", "FD002": "s", "FD003": "^", "FD004": "D"}
for ax, cu in zip(axs, (2, 10)):
    kap = cu + 1; tau = 1 / kap
    for f in FL:
        rho = SU[("mlp", f)]["ratios"]; pop = np.sort(rho[rng.integers(0, len(rho), 600000)] * np.exp(0.08 * rng.normal(size=600000))); cp = np.concatenate([[0], np.cumsum(pop)]); MP = len(pop)
        def Cq(q): j = np.searchsorted(pop, q); return (cu * (q * j - cp[j]) + ((cp[-1] - cp[j]) - q * (MP - j))) / MP
        qs = pop[int(tau * MP)]; h = 0.02 * qs; g = (np.searchsorted(pop, qs + h) - np.searchsorted(pop, qs - h)) / (2 * h * MP); c0 = Cq(qs); const = kap * tau * (1 - tau) / (2 * g)
        means, ses = [], []
        for n in NS:
            k = min(max(int(np.ceil((n + 1) * tau)), 1), n); draws = pop[rng.integers(0, MP, (3000, n))]; q = np.partition(draws, k - 1, axis=1)[:, k - 1]; ex = np.array([Cq(x) - c0 for x in q]) * n; means.append(ex.mean()); ses.append(ex.std() / np.sqrt(len(ex)))
        means, ses = np.array(means), np.array(ses); STATS8[(cu, f)] = (float(const), float(means[-1]), float(2 * ses[-1])); ax.fill_between(NS, means - 2 * ses, means + 2 * ses, color=fcol[f], alpha=0.18, lw=0); ax.plot(NS, means, color=fcol[f], marker=fmk[f], ms=4, lw=1.4, mec="white", mew=0.4, label=f)
        ax.axhline(const, color=fcol[f], lw=0.9, ls=":")
    ax.set_xscale("log"); clean_log(ax, [10, 100]); ax.grid(True, which="major"); ax.set_xlabel("sample size $n$"); panel(ax, f"({'ab'[(2, 10).index(cu)]}) $c_u/c_o={cu}$")
axs[0].set_ylabel("$n\\times$ excess cost (units of $p$)"); fig.legend(*axs[0].get_legend_handles_labels(), loc="outside upper center", ncol=4, handlelength=1.8, columnspacing=1.2)
save(fig, "fig_excess_scaling"); json.dump({f"{k[0]}_{k[1]}": v for k, v in STATS8.items()}, open(SRC + "fig8_stats.json", "w"))

# ====================================================================================================================
# Fig 9: label scarcity: RMSE of data-only / labelled-only constraint / unlabelled constraint, and the decision cost difference
# ====================================================================================================================
fig = plt.figure(figsize=(7.53, 2.9)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15]); nl = [5, 10, 20, 40]
rm = {"FD001": {"data": [24.24, 19.42, 17.61, 16.58], "kin_lab": [25.52, 19.66, 17.81, 17.58], "kin_unlab": [22.65, 18.81, 16.66, 17.26]}, "FD003": {"data": [26.52, 24.34, 16.13, 14.63], "kin_lab": [27.25, 24.83, 16.79, 16.17], "kin_unlab": [19.91, 20.43, 15.90, 15.63]}}   # semisup_FD001.log, semisup_FD003.log
for j, f in enumerate(("FD001", "FD003")):
    ax = fig.add_subplot(gs[0, j])
    for key, col, ls, mk, lab in (("data", GREY, "--", "v", "data only"), ("kin_lab", ORANGE, "-.", "s", "constraint, labelled only"), ("kin_unlab", BLUE, "-", "o", "constraint, + unlabelled")):
        ax.plot(range(4), rm[f][key], color=col, ls=ls, lw=2.0 if key == "kin_unlab" else 1.4, marker=mk, ms=5, mec="white", mew=0.5, label=lab)
    ax.fill_between(range(4), rm[f]["kin_unlab"], rm[f]["data"], where=[a < b for a, b in zip(rm[f]["kin_unlab"], rm[f]["data"])], color=BLUE, alpha=0.15, lw=0)
    ax.fill_between(range(4), rm[f]["kin_unlab"], rm[f]["data"], where=[a > b for a, b in zip(rm[f]["kin_unlab"], rm[f]["data"])], color=ORANGE, alpha=0.18, lw=0)
    ax.set_xticks(range(4)); ax.set_xticklabels([str(v) for v in nl]); ax.set_xlabel("labelled engines $n_{\\mathrm{lab}}$"); panel(ax, f"({'ab'[j]}) {f}"); ax.grid(True, axis="y")
    if j == 0: ax.set_ylabel("held-out RMSE (cycles)")
ax = fig.add_subplot(gs[0, 2]); ax.axhspan(0, 0.2, color="#f6e7e0", lw=0, zorder=0); ax.axhspan(-0.2, 0, color="#e1eefb", lw=0, zorder=0)
d = {"FD001": ([5, 10, 20], [-.002, .049, .002], [-.061, .009, -.040], [.054, .091, .043]), "FD003": ([10, 20], [.056, .043], [.013, .009], [.103, .078])}   # results_fleet_ss*_FD00x.txt, c_u/c_o=5
for i, (k, (x, m, lo, hi)) in enumerate(d.items()):
    c = BLUE if i == 0 else ORANGE; xx = np.array([{5: 0, 10: 1, 20: 2}[v] for v in x]) + (-0.08 if i == 0 else 0.08); m, lo, hi = map(np.array, (m, lo, hi))
    ax.errorbar(xx, m, yerr=[m - lo, hi - m], color=c, marker="o" if i == 0 else "s", ms=5, lw=1.4, capsize=3, capthick=1.0, mec="white", mew=0.6, zorder=3, label=k)
ax.axhline(0, color="#52514e", lw=0.8); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["5", "10", "20"]); ax.set_xlim(-0.4, 2.4); ax.set_ylim(-0.085, 0.13); ax.set_xlabel("labelled engines $n_{\\mathrm{lab}}$"); ax.set_ylabel("extra cost / $P_0$"); panel(ax, "(c) decision cost"); ax.grid(True, axis="y")
ax.text(2.38, 0.122, "worse", ha="right", va="top", color="#a14a2a"); ax.text(2.38, -0.078, "better", ha="right", va="bottom", color="#2a5d9a"); ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.78), handlelength=1.4, borderaxespad=0.2)
fig.legend(handles=[Line2D([], [], color=GREY, lw=1.4, ls="--", marker="v", ms=5, label="data only"), Line2D([], [], color=ORANGE, lw=1.4, ls="-.", marker="s", ms=5, label="constraint, labelled only"), Line2D([], [], color=BLUE, lw=2.0, marker="o", ms=5, label="constraint, + unlabelled")], loc="outside upper center", ncol=3, handlelength=2.0, columnspacing=1.5)
save(fig, "fig_label_scarcity")
print("ok")
