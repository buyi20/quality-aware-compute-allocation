# Quality-Aware Compute Allocation for LLM Pre-training

Code and data for the manuscript

> **A Data-Quality-Aware Scaling Model with Cost Constraints for Compute Allocation in Large Language Model Pre-training**
> Yaya Liu, Lianshuang Zhang and Xin Zhao, Business School, University of Shanghai for Science and Technology.

The paper allocates a fixed pre-training compute budget among model size $N$, training tokens $D$ and data quality $Q$, where raising quality costs compute. The loss is a separable scaling law whose reducible terms are multiplied by a quality factor,

$$L(N,D,Q)=E+\left(aN^{-\alpha}+bD^{-\beta}\right)e^{-\tau(Q-Q_0)},\qquad kND+D\,h(Q)\le C,\quad Q_0\le Q\le\bar Q .$$

The binding budget reduces the problem to a strictly convex problem in $\ln N$ and a one-dimensional search over $Q$, which a branch-and-bound algorithm with safeguarded interval bounds solves with an optimality gap that is certified in exact arithmetic (Algorithm 1 of the paper).

## Repository layout

| Path | Content |
|---|---|
| `src/model.py` | Model inputs, inner problem with safeguarded bounds, Algorithm 1 (`Problem.solve`) |
| `src/run_experiments.py` | Computational experiments of Sections 6 and 7; writes `results/` |
| `src/make_figures.py`, `src/nstyle.py` | Data figures (Figs. 3–6) from `results/` and `data/`; writes `figures/` |
| `src/quality_coordinate.py` | Quality coordinate of Eq. (21) and baseline quality $Q_0$ from Meta-rater document ratings |
| `results/` | Outputs of `run_experiments.py` reported in the paper (allocations, transitions, method comparison, sensitivity) |
| `data/quality_domain_summary.csv` | Domain-level quality scores (reference sample, check documents, all documents) |

## Data sources

* **Scale parameters** $E=1.69$, $a=406.4$, $b=410.7$, $\alpha=0.34$, $\beta=0.28$: published parametric fit of Hoffmann et al. (2022), *An Empirical Analysis of Compute-Optimal Large Language Model Training*, NeurIPS 35.
* **Document quality ratings**: the public [SlimPajama-Meta-rater](https://huggingface.co/datasets/opendatalab/SlimPajama-Meta-rater) dataset (Zhuang et al., ACL 2025). The study uses 261,086 distinct documents, namely a reference sample of 51,230 documents from all seven SlimPajama domains, which fixes the scale of the coordinate, and 209,856 further ArXiv and GitHub documents. The raw documents are not redistributed here and can be downloaded from the dataset page. `data/quality_domain_summary.csv` contains the resulting domain scores, and $Q_0=0.543$ is their equal-domain mean over all distinct documents.
* All other inputs (quality benefit $\tau$, attainable bound $\bar Q$, attention coefficient, processing-cost shapes and scale) are scenario inputs stated in the paper and in `src/model.py`.

## Reproducing the results

```bash
pip install -r requirements.txt
cd src
python run_experiments.py      # regenerates results/
python make_figures.py         # writes figures/Fig3.pdf to figures/Fig6.pdf
```

Single allocation:

```python
from model import Problem
sol = Problem(C=1e19, shape='exp').solve()   # tol = 1e-6
print(sol['N'], sol['D'], sol['Q'], sol['gap'])
```

Quality coordinate from Meta-rater JSON Lines files (records carry the Meta-rater signal fields, an `id` and a SlimPajama domain):

```bash
python quality_coordinate.py --reference reference/*.jsonl.xz --check check/*.jsonl.xz --out ../data
```

The results in the paper were computed with Python 3.12, NumPy 2.5, SciPy 1.18, pandas 3.0 and Matplotlib 3.11 on a laptop CPU (Intel Core i7-14650HX). Runtimes in `results/bench_*.csv` are wall-clock times on that machine.

## Note on the optimality gap

Eq. (17) and Proposition 2 hold in exact arithmetic. The implementation evaluates them in double precision and subtracts a rounding allowance of $8u_{\mathrm{mach}}F$, so the reported gaps are numerical estimates of the exact-arithmetic bounds.

## Citation

If you use this code, please cite the paper (citation details will be added after publication):

```bibtex
@unpublished{liu2026qualityaware,
  author = {Liu, Yaya and Zhang, Lianshuang and Zhao, Xin},
  title  = {A Data-Quality-Aware Scaling Model with Cost Constraints for Compute Allocation in Large Language Model Pre-training},
  note   = {Manuscript},
  year   = {2026}
}
```

## License

Code and data in this repository are released under the MIT License (see `LICENSE`). The SlimPajama-Meta-rater ratings remain subject to their original license.

Contact: Xin Zhao, zhaoxin07593@usst.edu.cn
