"""All computational experiments of Sections 6 and 7. Run: python run_experiments.py  (writes ../results/)."""
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

from model import (Problem, state, local_threshold_budget, g, Q0, K, ALPHA, BETA, E, TAU_REF,
                   QBAR_REF, SHAPES, ETA, CTX)

OUT = Path(__file__).resolve().parent.parent / 'results'
OUT.mkdir(exist_ok=True)
BUDGETS = np.geomspace(1e16, 1e25, 181)
MAIN_BUDGETS = [1e19, 1e22, 1e24]
summary = {}


def certified(C, shape='exp', rho=1.0, tau=TAU_REF, qbar=QBAR_REF, tol=1e-6):
    return Problem(C, shape, rho, tau, qbar).solve(tol=tol)


def transition_budgets(shape, rho=1.0, tau=TAU_REF, qbar=QBAR_REF, lo=13.0, hi=27.0):
    """Budgets at which optimal quality leaves Q0 and reaches qbar (bisection on log10 C)."""
    def leaves(lc):
        return certified(10 ** lc, shape, rho, tau, qbar, tol=1e-7)['Q'] > Q0 + 1e-6

    def reaches(lc):
        return certified(10 ** lc, shape, rho, tau, qbar, tol=1e-7)['Q'] > qbar - 1e-6

    out = {}
    for name, pred in (('leave', leaves), ('reach', reaches)):
        a, b = lo, hi
        assert not pred(a) and pred(b), (shape, rho, tau, qbar, name)
        for _ in range(40):
            m = 0.5 * (a + b)
            a, b = (a, m) if pred(m) else (m, b)
        out[name] = 10 ** (0.5 * (a + b))
    out['type'] = 'jump' if out['reach'] / out['leave'] < 1 + 1e-6 else 'continuous'
    return out


# ------------------------------------------------------------------ 1. main allocations
rows = []
for shape in SHAPES:
    for C in MAIN_BUDGETS:
        p = Problem(C, shape)
        r = p.solve()
        base = p.baseline()
        r['reducible_reduction_pct'] = 100 * (base - r['U']) / base
        r['tokens_per_param'] = r['D'] / r['N']
        rows.append(r)
main = pd.DataFrame(rows)
main.to_csv(OUT / 'main_allocations.csv', index=False)
summary['max_budget_residual'] = float(main.budget_residual.max())
summary['max_balance_residual'] = float(main.balance_residual.max())
print(main[['shape', 'C', 'N', 'D', 'Q', 'loss', 'train_share', 'quality_share', 'attn_share',
            'reducible_reduction_pct', 'gap']].to_string(), flush=True)

# ------------------------------------------------------------------ 2. budget paths and transitions
path = pd.DataFrame([certified(C, shape) for shape in SHAPES for C in BUDGETS])
path.to_csv(OUT / 'budget_path.csv', index=False)
trans = []
for shape in SHAPES:
    t = transition_budgets(shape)
    t['shape'] = shape
    t['local_threshold'] = local_threshold_budget(shape)
    trans.append(t)
trans = pd.DataFrame(trans)
trans.to_csv(OUT / 'transitions.csv', index=False)
print(trans.to_string(), flush=True)

# outer profiles around the logarithmic-cost jump
cj = float(trans.loc[trans['shape'] == 'log', 'leave'].iloc[0])
qs = np.linspace(Q0, QBAR_REF, 401)
prof = []
for fac in (0.8, 1.0, 1.25):
    p = Problem(cj * fac, 'log')
    v0 = p.V(Q0)
    for q in qs:
        prof.append(dict(factor=fac, C=cj * fac, Q=q, relative=p.V(float(q)) - v0))
prof = pd.DataFrame(prof)
prof.to_csv(OUT / 'jump_profiles.csv', index=False)
at = prof[prof.factor == 1.0].reset_index(drop=True)
interior = at.iloc[1:-1]
summary['log_jump_interior_stationary_Q'] = float(interior.loc[interior.relative.idxmax(), 'Q'])
summary['log_jump_endpoint_difference'] = float(at.relative.iloc[-1])

# ------------------------------------------------------------------ 3. solver benchmark (543 cases)
def reducible(z, p):
    x, q = float(z[0]), float(min(max(z[1], Q0), p.qbar))
    hb = p.h(q) / 1e9
    return math.exp(-p.tau * (q - Q0)) * p.F(x, hb)


bench = []
for shape in SHAPES:
    for C in BUDGETS:
        p = Problem(C, shape)
        r = p.solve()
        U = r['U']
        bench.append(dict(C=C, shape=shape, method='Certified branch-and-bound', excess=0.0,
                          evals=r['calls'], seconds=r['seconds'], gap=r['gap'], Q=r['Q'], state=state(r)))
        x0 = Problem(C, shape).inner(Q0)['x']
        bounds = [(-15, 15), (Q0, QBAR_REF)]
        # local search from baseline quality
        t0 = time.perf_counter()
        res = minimize(reducible, [x0, Q0], args=(p,), method='L-BFGS-B', bounds=bounds)
        bench.append(dict(C=C, shape=shape, method='Local search from baseline', excess=res.fun - U,
                          evals=res.nfev, seconds=time.perf_counter() - t0))
        # ten-start local search
        t0 = time.perf_counter(); best = math.inf; nf = 0
        for q in np.linspace(Q0, QBAR_REF, 10):
            res = minimize(reducible, [x0, q], args=(p,), method='L-BFGS-B', bounds=bounds)
            best = min(best, res.fun); nf += res.nfev
        bench.append(dict(C=C, shape=shape, method='Multi-start local search (10)', excess=best - U,
                          evals=nf, seconds=time.perf_counter() - t0))
        # uniform quality grids with exact inner solves
        for ng in (41, 81):
            t0 = time.perf_counter(); pg = Problem(C, shape)
            best = min(pg.V(float(q)) for q in np.linspace(Q0, QBAR_REF, ng))
            bench.append(dict(C=C, shape=shape, method=f'Uniform Q grid ({ng})', excess=best - U,
                              evals=ng, seconds=time.perf_counter() - t0))
        # differential evolution, five seeds
        for seed in range(5):
            t0 = time.perf_counter()
            res = differential_evolution(reducible, bounds, args=(p,), seed=seed)
            bench.append(dict(C=C, shape=shape, method='Differential evolution (5 seeds)',
                              excess=res.fun - U, evals=res.nfev, seconds=time.perf_counter() - t0, seed=seed))
    print('benchmark done for', shape, flush=True)
bench = pd.DataFrame(bench)
bench.to_csv(OUT / 'bench_raw.csv', index=False)
order = ['Certified branch-and-bound', 'Local search from baseline', 'Multi-start local search (10)',
         'Uniform Q grid (41)', 'Uniform Q grid (81)', 'Differential evolution (5 seeds)']
tab = []
for m in order:
    s = bench[bench.method == m]
    fail = s[s.excess > 1e-5]
    tab.append(dict(method=m, runs=len(s), failures=len(fail), max_excess=float(s.excess.max()),
                    median_evals=float(s.evals.median()), median_ms=1000 * float(s.seconds.median()),
                    fail_C_min=float(fail.C.min()) if len(fail) else None,
                    fail_C_max=float(fail.C.max()) if len(fail) else None,
                    fail_shapes=','.join(sorted(set(fail['shape']))) if len(fail) else ''))
tab = pd.DataFrame(tab)
tab.to_csv(OUT / 'bench_summary.csv', index=False)
cb = bench[bench.method == order[0]]
summary['bench_max_gap'] = float(cb.gap.max())
summary['bench_interior_cases'] = int((cb.state == 1).sum())
summary['bench_interior_median_inner'] = float(cb[cb.state == 1].evals.median())
print(tab.to_string(), flush=True)

# ------------------------------------------------------------------ 4. benefit and cost-scale map at 1e19 (exp)
taus = np.linspace(-0.2, 1.0, 25)
rhos = np.geomspace(0.1, 10, 33)
heat = pd.DataFrame([dict(tau=t, rho=r, **{k: v for k, v in certified(1e19, 'exp', r, t).items()
                                           if k in ('Q', 'gap')}) for t in taus for r in rhos])
heat.to_csv(OUT / 'regime_heatmap.csv', index=False)


def rho_threshold(tau, which):
    """Largest rho at 1e19 (exp) for which quality leaves Q0 ('any') or reaches the bound ('bound')."""
    def ok(lr):
        q = certified(1e19, 'exp', 10 ** lr, tau, tol=1e-7)['Q']
        return q > Q0 + 1e-6 if which == 'any' else q > QBAR_REF - 1e-6
    a, b = -3.0, 3.0
    if not ok(a):
        return None
    if ok(b):
        return math.inf
    for _ in range(40):
        m = 0.5 * (a + b)
        a, b = (m, b) if ok(m) else (a, m)
    return 10 ** a


summary['rho_thresholds'] = {str(t): dict(any=rho_threshold(t, 'any'), bound=rho_threshold(t, 'bound'))
                             for t in (0.25, 0.5, 1.0)}

# ------------------------------------------------------------------ 5. cost scale and transition budgets
kappa = (ALPHA + BETA) / BETA
tv = []
for shape in SHAPES:
    inc1 = g(1.0, shape) - g(Q0, shape)
    for r in np.geomspace(0.1, 10, 21):
        t = transition_budgets(shape, rho=r)
        tv.append(dict(shape=shape, rho=r, increment=r * inc1, entry=t['leave'], saturation=t['reach'],
                       type=t['type']))
tv = pd.DataFrame(tv)
tv.to_csv(OUT / 'transition_vs_rho.csv', index=False)
slopes = {}
for shape in SHAPES:
    s = tv[tv['shape'] == shape]
    slopes[shape] = dict(entry=float(np.polyfit(np.log(s.rho), np.log(s.entry), 1)[0]),
                         saturation=float(np.polyfit(np.log(s.rho), np.log(s.saturation), 1)[0]),
                         types=sorted(set(s['type'])))
summary['kappa'] = kappa
summary['transition_slopes'] = slopes
matched = {}
for shape in SHAPES:
    r = 2e9 / (g(1.0, shape) - g(Q0, shape))
    matched[shape] = transition_budgets(shape, rho=r)['leave']
summary['matched_endpoint_entry'] = matched

# ------------------------------------------------------------------ 6. attainable improvement
wt = []
for w in (0.1, 0.2, 0.3, 0.4, 1 - Q0):
    qbar = Q0 + w
    row = dict(w=w)
    for shape in ('exp', 'power'):
        t = transition_budgets(shape, qbar=qbar)
        row[f'{shape}_leave'], row[f'{shape}_reach'] = t['leave'], t['reach']
    t = transition_budgets('log', qbar=qbar)
    row['log_jump'], row['log_type'] = t['leave'], t['type']
    row['local_over_jump'] = local_threshold_budget('log') / t['leave']
    p = Problem(1e22, 'exp', qbar=qbar)
    r = p.solve()
    row['reduction_1e22_pct'] = 100 * (p.baseline() - r['U']) / p.baseline()
    wt.append(row)
wt = pd.DataFrame(wt)
wt.to_csv(OUT / 'width.csv', index=False)
print(wt.to_string(), flush=True)

# ------------------------------------------------------------------ 7. robust allocation
TAU_MIN, RHO_MAX = TAU_REF / 2, 2.0
rob = []
for C in MAIN_BUDGETS:
    r = certified(C, 'exp', RHO_MAX, TAU_MIN)
    r['state'] = state(r)
    rob.append(r)
rob = pd.DataFrame(rob)
rob.to_csv(OUT / 'robust.csv', index=False)
summary['robust'] = rob[['C', 'Q', 'quality_share', 'loss']].to_dict('records')

summary['saturation_exponents'] = dict(N=BETA / (ALPHA + BETA), D=ALPHA / (ALPHA + BETA))
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')
print(json.dumps(summary, indent=2, default=float))
