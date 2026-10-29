"""Data figures of the paper (Figs. 3-6) from ../results and ../data. Run after run_experiments.py.

Outputs (../figures/): Fig3.pdf to Fig6.pdf, numbered as in the paper
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nstyle import *  # noqa: E402,F401  (style, palette, helpers)
from model import Q0, ALPHA, BETA, TAU_REF, g  # noqa: E402

R = HERE.parent / 'results'
FIG = HERE.parent / 'figures'
FIG.mkdir(exist_ok=True)
S = json.loads((R / 'summary.json').read_text(encoding='utf-8'))
trans = pd.read_csv(R / 'transitions.csv').set_index('shape')
C_JUMP = trans.loc['log', 'leave']
C_LOC = trans.loc['log', 'local_threshold']


def save_to(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('saved', name)


# ------------------------------------------------------------------ calibration inputs
dom = pd.read_csv(HERE.parent / 'data' / 'quality_domain_summary.csv')
full = dom[dom.set == 'all'].set_index('domain')
fig = plt.figure(figsize=(TEXTWIDTH * 0.85, 2.3))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.55)
ax = fig.add_subplot(gs[0, 0])
order = full.Q_mean.sort_values().index.tolist()
y = np.arange(len(order)); q = full.loc[order, 'Q_mean'].to_numpy()
ax.hlines(y, Q0, q, color=GREY, lw=0.8)
ax.scatter(q, y, s=22, color=CATEGORICAL[0], zorder=3, lw=0)
for yi, v in zip(y, q):
    ax.text(v + (0.015 if v >= Q0 else -0.015), yi, f'{v:.2f}', va='center', ha='left' if v >= Q0 else 'right', fontsize=6.5)
ax.axvline(Q0, color=ACCENT_RED, lw=0.9, ls='--')
ax.text(Q0 + 0.012, len(order) - 0.35, f'$Q_0$ = {Q0:.3f}', color=ACCENT_RED, fontsize=6.8, ha='left', va='bottom')
ax.set_yticks(y, [DOMAIN[d] for d in order]); ax.tick_params(axis='y', length=0)
ax.set_xlim(0.2, 0.82); ax.set_ylim(-0.6, len(order) + 0.2)
ax.set_xlabel('Domain quality $Q_d$')
label(ax, 'a', x=-0.52)
ax = fig.add_subplot(gs[0, 1])
qs = np.linspace(Q0, 1, 200)
for k in ('exp', 'power', 'log'):
    ax.plot(qs, [(g(v, k) - g(Q0, k)) / 1e9 for v in qs], color=COST[k], ls=LS[k], **LINE, label=COST_LABEL[k])
ax.set_xlabel('Quality $Q$'); ax.set_ylabel(r'$h(Q)$ ($10^9$ FLOPs per token)')
ax.legend(loc='upper left', fontsize=6.8, handlelength=2)
label(ax, 'b', x=-0.3)
save_to(fig, 'Fig3')

# ------------------------------------------------------------------ solver benchmark
d = pd.read_csv(R / 'bench_raw.csv')
STY = {'Local search from baseline': (CATEGORICAL[0], 'o', 'Local from $Q_0$'),
       'Differential evolution (5 seeds)': (CATEGORICAL[3], 's', 'Diff. evolution'),
       'Uniform Q grid (41)': (CATEGORICAL[2], '^', 'Q-grid (41)'),
       'Uniform Q grid (81)': (CATEGORICAL[2], 'v', 'Q-grid (81)'),
       'Multi-start local search (10)': (CATEGORICAL[4], 'P', 'Multi-start (10)'),
       'Certified branch-and-bound': (ACCENT_RED, '*', 'Certified branch-and-bound')}
BB_LABEL_B = 'Certified branch-and-bound\n(largest reported gap)'
fig = plt.figure(figsize=(TEXTWIDTH * 0.9, 2.35))
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.4)
ax = fig.add_subplot(gs[0, 0])
lg = d[d['shape'] == 'log']
ax.axvspan(C_JUMP, C_LOC, color=LIGHT_BLUE, lw=0, alpha=0.7)
top = max(0.02, float(lg.excess.max()) * 1.15)
ax.text(np.sqrt(C_JUMP * C_LOC), top * 0.97, 'jump to local\nthreshold', fontsize=6.5, ha='center', va='top')
for m in ('Local search from baseline', 'Differential evolution (5 seeds)', 'Certified branch-and-bound'):
    s = lg[lg.method == m].groupby('C').excess.max().clip(lower=0)
    c, mk, lab = STY[m]
    ax.plot(s.index, s.values, color=c, marker=None if m.startswith('Cert') else mk, ms=2.4, mew=0, lw=1.1,
            label=lab + (' (worst of 5)' if m.startswith('Diff') else ''))
ax.set_xscale('log'); ax.set_xlim(1e16, 1e25); ax.set_ylim(-0.02 * top, top)
ax.set_xlabel('Compute budget $C$ (FLOPs)'); ax.set_ylabel('Excess loss')
ax.legend(loc='center right', fontsize=6.2, borderaxespad=0.2)
label(ax, 'a', x=-0.17)
ax = fig.add_subplot(gs[0, 1])
for m, (c, mk, lab) in STY.items():
    s = d[d.method == m]
    t = s.seconds.median() * 1e3
    cert = m.startswith('Cert')
    r = float(s.gap.max()) if cert else float(s.excess.max())
    ax.scatter(t, r, s=34 if cert else 20, marker=mk, color=c, edgecolor='none', zorder=3)
    ax.annotate(BB_LABEL_B if cert else lab, (t, r), xytext=(4, -10 if cert else 4), textcoords='offset points', fontsize=6.3)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-11, 10); ax.set_xlim(0.3, 30)
ax.set_xticks([0.5, 1, 2, 5, 10, 20], ['0.5', '1', '2', '5', '10', '20']); ax.minorticks_off()
ax.axhline(1e-5, color=GREY, lw=0.6, ls=':')
ax.text(29, 1.4e-5, 'failure threshold', fontsize=6.2, color=GREY, ha='right', va='bottom')
ax.set_xlabel('Median time per case (ms)'); ax.set_ylabel('Maximum excess loss or reported gap')
label(ax, 'b', x=-0.22)
save_to(fig, 'Fig5')

# ------------------------------------------------------------------ allocation paths
bp = pd.read_csv(R / 'budget_path.csv')
jp = pd.read_csv(R / 'jump_profiles.csv')
fig = plt.figure(figsize=(TEXTWIDTH, 2.45))
gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 1], wspace=0.42)
ax = fig.add_subplot(gs[0, 0])
ax.axvspan(C_JUMP, C_LOC, color=ACCENT_RED, alpha=0.07, lw=0)
for k in ('exp', 'power', 'log'):
    s = bp[bp['shape'] == k].sort_values('C')
    jumps = np.where(np.abs(np.diff(s.Q.to_numpy())) > 0.2)[0]
    if len(jumps):
        cut = jumps[0] + 1
        ax.plot(s.C.iloc[:cut], s.Q.iloc[:cut], color=COST[k], ls=LS[k], **LINE, label=COST_LABEL[k])
        ax.plot(s.C.iloc[cut:], s.Q.iloc[cut:], color=COST[k], ls=LS[k], **LINE)
        cj = trans.loc[k, 'leave']
        ax.plot([cj, cj], [Q0, 1.0], color=COST[k], lw=0.6, ls=':')
    else:
        ax.plot(s.C, s.Q, color=COST[k], ls=LS[k], **LINE, label=COST_LABEL[k])
    ax.plot(trans.loc[k, 'leave'], Q0 if trans.loc[k, 'type'] == 'continuous' else 1.0, marker='o', color=COST[k], **MARK)
    ax.plot(trans.loc[k, 'reach'], 1.0, marker='o', color=COST[k], **MARK)
ax.axvline(C_LOC, color=ACCENT_RED, lw=0.7, ls=(0, (1, 1.5)))
ax.annotate('', xy=(C_JUMP, 0.49), xytext=(C_LOC, 0.49), arrowprops=dict(arrowstyle='<->', lw=0.7, color=ACCENT_RED))
ax.text(np.sqrt(C_JUMP * C_LOC), 0.505, f'{C_LOC / C_JUMP:.1f}×', ha='center', fontsize=7, color=ACCENT_RED)
ax.set_xscale('log'); ax.set_xlim(1e16, 1e25); ax.set_ylim(0.46, 1.05)
ax.set_xlabel('Compute budget $C$ (FLOPs)'); ax.set_ylabel(r'Optimal quality $Q^*$')
ax.legend(loc='center right', fontsize=6.8, handlelength=2)
label(ax, 'a', x=-0.2)
ax = fig.add_subplot(gs[0, 1])
s = bp[bp['shape'] == 'exp'].sort_values('C')
ax.stackplot(s.C, s.train_share * 100, s.quality_share * 100, s.attn_share * 100,
             colors=[LIGHT_BLUE, CATEGORICAL[2], CATEGORICAL[3]], lw=0, labels=['Training', 'Data processing', 'Attention'])
ax.set_xscale('log'); ax.set_xlim(1e16, 1e25); ax.set_ylim(0, 100)
ax.set_xlabel('Compute budget $C$ (FLOPs)'); ax.set_ylabel('Budget share (%)')
ax.legend(loc='lower left', fontsize=6.3, handlelength=1.2)
label(ax, 'b', x=-0.3)
ax = fig.add_subplot(gs[0, 2])
for (fac, gq), c in zip(jp.groupby('factor'), [CATEGORICAL[0], ACCENT_RED, GREY]):
    gq = gq.sort_values('Q')
    ax.plot(gq.Q, gq.relative * 1e3, color=c, lw=1.2, label=rf'{fac:g}$\,C^\dagger$')
ax.axhline(0, color=GREY, lw=0.5)
ax.set_xlabel('Quality $Q$'); ax.set_ylabel(r'$V_C(Q)-V_C(Q_0)$ ($\times10^{-3}$)')
ax.legend(loc='best', fontsize=6.6, handlelength=1.4)
label(ax, 'c', x=-0.3)
save_to(fig, 'Fig4')

# ------------------------------------------------------------------ sensitivity
hm = pd.read_csv(R / 'regime_heatmap.csv')
tv = pd.read_csv(R / 'transition_vs_rho.csv')
fig = plt.figure(figsize=(TEXTWIDTH * 0.9, 2.35))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.55)
ax = fig.add_subplot(gs[0, 0])
H = hm.pivot(index='tau', columns='rho', values='Q').sort_index()
taus, rhos = H.index.to_numpy(), H.columns.to_numpy()
lr = np.log10(rhos); dl, dt = lr[1] - lr[0], taus[1] - taus[0]
im = ax.imshow(H.to_numpy(), origin='lower', cmap=SEQ_CMAP, vmin=Q0, vmax=1.0, aspect='auto',
               extent=[lr[0] - dl / 2, lr[-1] + dl / 2, taus[0] - dt / 2, taus[-1] + dt / 2], interpolation='nearest')
ax.axhline(0, color=BLACK, lw=0.6)
ax.text(lr[0] + 0.05, -0.105, r'$\tau\leq0$: baseline optimal', fontsize=6.5, va='center', color=BLACK)
for t, ls in ((TAU_REF / 2, ':'), (TAU_REF, '--'), (TAU_REF * 2, ':')):
    ax.axhline(t, color=ACCENT_RED, lw=0.8, ls=ls)
    ax.text(lr[-1] + dl / 2 - 0.03, t + 0.012, rf'$\tau={t:g}$', fontsize=6.5, color=ACCENT_RED, ha='right', va='bottom')
ax.set_xticks([-1, 0, 1], ['0.1', '1', '10'])
ax.set_xlabel(r'Cost scale $\rho$'); ax.set_ylabel(r'Quality benefit $\tau$')
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04, ticks=[Q0, 0.75, 1.0]); cb.outline.set_visible(False)
cb.ax.set_yticklabels([r'$Q_0$', '0.75', '1.00'])
cb.ax.tick_params(labelsize=6.5, length=2); cb.set_label(r'$Q^*$ at $C=10^{19}$ FLOPs', fontsize=7)
label(ax, 'a', x=-0.3)
ax = fig.add_subplot(gs[0, 1])
for k in ('exp', 'power', 'log'):
    s = tv[tv['shape'] == k].sort_values('increment')
    x = s.increment / 1e9
    if (s['type'] == 'jump').all():
        ax.plot(x, s.entry, color=COST[k], marker='D', ms=2.6, mew=0, lw=1.2, label=f'{COST_LABEL[k]}: jump')
    else:
        ax.plot(x, s.entry, color=COST[k], marker='o', ms=2.6, mew=0, lw=1.2, label=f'{COST_LABEL[k]}: leaves $Q_0$')
        ax.plot(x, s.saturation, color=COST[k], ls='--', lw=1.0, label=rf'{COST_LABEL[k]}: reaches $\bar Q$')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'Endpoint increment $\rho\,[g(1)-g(Q_0)]$ ($10^9$ FLOPs/token)')
ax.set_ylabel('Transition budget (FLOPs)')
kap = (ALPHA + BETA) / BETA
x0, y0 = 2.0, 1e15
ax.plot([x0, x0 * 10], [y0, y0 * 10 ** kap], color=GREY, lw=0.8)
ax.text(x0 * 4.0, y0 * 10 ** (kap * 0.25), r'slope $(\alpha+\beta)/\beta=$' + f'{kap:.3f}', fontsize=6.3, color=GREY, ha='left', va='top')
ax.legend(loc='upper left', fontsize=6.0, handlelength=2.2, borderaxespad=0.2)
label(ax, 'b', x=-0.25)
save_to(fig, 'Fig6')
