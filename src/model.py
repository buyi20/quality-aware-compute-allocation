"""Quality-aware compute allocation model and certified solver (Algorithm 1 of the paper).

Units inside the code: parameters and tokens in billions (n, d), compute in 1e18 FLOPs (c).
"""
import heapq
import math
import time

from scipy.optimize import brentq

# ---------------------------------------------------------------- inputs
# Scale parameters: published parametric fit of Hoffmann et al. (2022), Eq. (1) of the paper.
E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28
A_B = A * 1e9 ** (-ALPHA)          # coefficient for n in billions
B_B = B * 1e9 ** (-BETA)           # coefficient for d in billions
# Reference quality of the available corpus (equal-domain mean of the Meta-rater quality coordinate).
Q0 = 0.5429272883586321
# Compute constants: k = 6 + eta * L_ctx.
ETA, CTX = 2e-4, 2048
K = 6 + ETA * CTX
# Reference scenario for the quality benefit and the attainable bound.
TAU_REF, QBAR_REF = 0.5, 1.0
EPS_MACH = 2.0 ** -52

SHAPES = ['exp', 'power', 'log']


def g(q, shape):
    """Processing-cost shape g(Q) in FLOPs per retained token."""
    if shape == 'exp':
        return 1e7 * math.exp(6 * q)
    if shape == 'power':
        return 5e9 * q ** 4
    if shape == 'log':
        return 2e9 * math.log1p(10 * q)
    raise ValueError(shape)


def g_prime(q, shape):
    if shape == 'exp':
        return 6e7 * math.exp(6 * q)
    if shape == 'power':
        return 2e10 * q ** 3
    if shape == 'log':
        return 2e10 / (1 + 10 * q)
    raise ValueError(shape)


class Problem:
    """One allocation problem, Eq. (11): budget C, cost shape and scale, benefit tau, bound Qbar."""

    def __init__(self, C, shape='exp', rho=1.0, tau=TAU_REF, qbar=QBAR_REF, k=K):
        self.C, self.c = float(C), float(C) / 1e18
        self.shape, self.rho, self.tau, self.qbar, self.k = shape, rho, tau, qbar, k
        self.cache = {}
        self.calls = 0

    # processing cost per retained token, FLOPs
    def h(self, q):
        return self.rho * (g(q, self.shape) - g(Q0, self.shape))

    # inner objective F_Q(x) and its derivative, Eq. (12), with h in units of 1e9 FLOPs/token
    def F(self, x, hb):
        return A_B * math.exp(-ALPHA * x) + B_B * self.c ** (-BETA) * (self.k * math.exp(x) + hb) ** BETA

    def dF(self, x, hb):
        z = self.k * math.exp(x)
        return -ALPHA * A_B * math.exp(-ALPHA * x) + BETA * B_B * self.c ** (-BETA) * (z + hb) ** (BETA - 1) * z

    def inner(self, q):
        """Safeguarded inner solve at quality q: returns x, upper value B_C, lower bound B_C^-."""
        if q in self.cache:
            return self.cache[q]
        self.calls += 1
        hb = self.h(q) / 1e9
        x = brentq(lambda t: self.dF(t, hb), -40.0, 40.0, xtol=1e-13, rtol=4 * EPS_MACH, maxiter=500)
        d = 1e-12
        xl, xu = x - d, x + d
        while not self.dF(xl, hb) < 0:
            xl -= d; d *= 2
        d = 1e-12
        while not self.dF(xu, hb) > 0:
            xu += d; d *= 2
        xm = 0.5 * (xl + xu)
        Fm = self.F(xm, hb)
        M = max(abs(self.dF(xl, hb)), abs(self.dF(xu, hb)))
        lower = Fm - M * (xu - xl) - 8 * EPS_MACH * Fm          # Eq. (15) with rounding allowance
        out = dict(x=xm, B=Fm, Bm=lower, hb=hb, V=math.exp(-self.tau * (q - Q0)) * Fm)
        self.cache[q] = out
        return out

    def V(self, q):
        return self.inner(q)['V']

    def lb(self, l, u):
        """Interval lower bound of Proposition 2, Eq. (14)."""
        Bm = self.inner(l)['Bm']
        if Bm <= 0:
            return -math.inf
        return min(math.exp(-self.tau * (l - Q0)), math.exp(-self.tau * (u - Q0))) * Bm

    def solve(self, tol=1e-6, ngrid=41):
        """Algorithm 1: certified epsilon-optimal allocation."""
        t0 = time.perf_counter()
        if self.tau <= 0 or self.qbar <= Q0:
            q = Q0
            U = self.V(q)
            gap = U - math.exp(-self.tau * 0.0) * self.inner(q)['Bm']
            nodes = 0
        else:
            grid = [Q0 + (self.qbar - Q0) * i / (ngrid - 1) for i in range(ngrid)]
            q = min(grid, key=self.V)
            U = self.V(q)
            heap = [(self.lb(Q0, self.qbar), Q0, self.qbar)]
            nodes = 0
            while heap and U - heap[0][0] > tol:
                _, l, u = heapq.heappop(heap)
                m = 0.5 * (l + u)
                v = self.V(m)
                if v < U:
                    U, q = v, m
                for lo, hi in ((l, m), (m, u)):
                    b = self.lb(lo, hi)
                    if b < U:
                        heapq.heappush(heap, (b, lo, hi))
                nodes += 1
                if nodes > 2_000_000:
                    raise RuntimeError('branch-and-bound did not converge')
            ell = U if not heap else min(U, heap[0][0])
            gap = U - ell
        r = self.inner(q)
        n = math.exp(r['x'])
        d = self.c / (self.k * n + r['hb'])
        N, D = n * 1e9, d * 1e9
        h = self.h(q)
        return dict(C=self.C, shape=self.shape, rho=self.rho, tau=self.tau, qbar=self.qbar,
                    Q=q, N=N, D=D, U=U, loss=E + U, gap=gap, calls=self.calls, nodes=nodes,
                    seconds=time.perf_counter() - t0,
                    train_share=6 * N * D / self.C, quality_share=D * h / self.C,
                    attn_share=ETA * CTX * N * D / self.C,
                    budget_residual=abs((self.k * N * D + D * h) / self.C - 1),
                    balance_residual=abs(ALPHA * A * N ** -ALPHA
                                         - BETA * B * D ** -BETA * self.k * N / (self.k * N + h))
                                     / (ALPHA * A * N ** -ALPHA))

    def baseline(self):
        """Reducible loss with parameters and tokens re-optimized at baseline quality."""
        return self.inner(Q0)['B']


def state(sol, qbar=None, tol=1e-6):
    qbar = sol['qbar'] if qbar is None else qbar
    if sol['Q'] - Q0 < tol:
        return 0
    if qbar - sol['Q'] < tol:
        return 2
    return 1


def local_threshold_budget(shape, rho=1.0, tau=TAU_REF):
    """Budget at which the local criterion of Eq. (17) first favours quality investment at Q0."""
    n_loc = ALPHA * BETA * rho * g_prime(Q0, shape) / (tau * (ALPHA + BETA) * K) / 1e9  # billions
    lo, hi = 10.0, 30.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        n = math.exp(Problem(10 ** mid, shape, rho, tau).inner(Q0)['x'])
        lo, hi = (mid, hi) if n < n_loc else (lo, mid)
    return 10 ** (0.5 * (lo + hi))
