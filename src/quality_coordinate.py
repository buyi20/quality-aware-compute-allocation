"""Quality coordinate of Section 5.2 (Eq. 19) from SlimPajama-Meta-rater document ratings.

Input: JSON Lines files (optionally .xz-compressed) whose records carry the Meta-rater quality
signals listed in FIELDS, a document id and a SlimPajama domain. The reference files define the
scale (percentile ranks and saturation points); the check files are scored with the unchanged scale
to test whether the construction carries over to new documents. The baseline quality Q0 is the
equal-domain mean of the domain scores over all distinct documents.

Usage:
    python quality_coordinate.py --reference ref/*.jsonl.xz --check check/*.jsonl.xz --out ../data
"""
import argparse
import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = {
    'education': ['fineweb_edu', 'modernbert_professionalism', 'modernbert_reasoning', 'qurater'],
    'readability_cleanliness': ['fluency_en', 'modernbert_readability', 'modernbert_cleanliness', 'ad_en'],
    'reference_similarity': ['dsir_books', 'dsir_wiki', 'dsir_math'],
    'text_statistics': ['rps_doc_word_count', 'rps_doc_num_sentences', 'rps_doc_unigram_entropy',
                        'rps_doc_frac_unique_words', 'rps_doc_frac_no_alph_words', 'rps_doc_frac_chars_top_2gram',
                        'rps_doc_frac_chars_top_3gram', 'rps_lines_uppercase_letter_fraction',
                        'rps_lines_ending_with_terminal_punctution_mark', 'rps_lines_numerical_chars_fraction',
                        'rps_doc_mean_word_length'],
}
FIELDS = sum(GROUPS.values(), [])          # 22 signals (the four QuRating criteria are averaged)
NEGATIVE = ['rps_doc_frac_no_alph_words', 'rps_doc_frac_chars_top_2gram', 'rps_doc_frac_chars_top_3gram',
            'rps_lines_uppercase_letter_fraction', 'rps_lines_numerical_chars_fraction']
DOMAINS = ['arxiv', 'book', 'c4', 'commoncrawl', 'github', 'stackexchange', 'wikipedia']
EPS = 0.01                                 # shift epsilon of Eq. (19)


def softmax(v):
    a = np.asarray(v, dtype=float)
    a = np.exp(a - a.max())
    return a / a.sum()


def read_records(path, domain_field):
    opener = lzma.open if str(path).endswith('.xz') else open
    fallback = next((d for d in DOMAINS if d in Path(path).name.lower()), None)
    rows = []
    with opener(path, 'rt', encoding='utf-8') as h:
        for line in h:
            x = json.loads(line)
            r = {'domain': x.get(domain_field) or fallback, 'id': str(x['id'])}
            for k in FIELDS:
                v = x.get(k)
                if v is None:
                    r[k] = np.nan
                elif k.startswith('modernbert_'):           # six-level rater: expected grade / 5
                    r[k] = float(softmax(v) @ np.arange(6) / 5)
                elif k in ('fluency_en', 'ad_en'):          # binary rater: favourable-class probability
                    r[k] = float(softmax(v)[1])
                elif k == 'fineweb_edu':
                    r[k] = float(v[0])
                elif k == 'qurater':                        # four criteria, kept separately for ranking
                    for j, t in enumerate(v):
                        r[f'qurater_{j}'] = float(t)
                    r[k] = float(np.mean(v))
                else:
                    r[k] = float(v)
            rows.append(r)
    return rows


def ref_cdf(ref, w, x):
    """Domain-weighted mid-rank percentile of x within the reference values."""
    ok = np.isfinite(ref)
    r, v = ref[ok], w[ok]
    o = np.argsort(r)
    r, v = r[o], v[o]
    unique, ix = np.unique(r, return_index=True)
    mass = np.add.reduceat(v, ix)
    prob = (np.cumsum(mass) - mass / 2) / mass.sum()
    if len(unique) == 1:
        return np.full(len(x), 0.5)
    return np.interp(x, unique, prob, left=0, right=1)


def weighted_quantile(ref, w, q):
    ok = np.isfinite(ref)
    r, v = ref[ok], w[ok]
    o = np.argsort(r)
    r, v = r[o], v[o]
    return np.interp(q, (np.cumsum(v) - v / 2) / v.sum(), r)


def smoothed_geometric_mean(G, weights=None):
    """Eq. (19): shift each dimension score by EPS, take the weighted geometric mean, rescale to [0, 1]."""
    w = np.full(G.shape[1], 1 / G.shape[1]) if weights is None else np.asarray(weights)
    return (np.exp(np.log(EPS + (1 - EPS) * G) @ w) - EPS) / (1 - EPS)


def score(df, is_ref):
    w = (1 / df.loc[is_ref, 'domain'].map(df.loc[is_ref, 'domain'].value_counts())).to_numpy()  # equal domain weight
    # missing values are imputed with the reference median
    for col in FIELDS + [f'qurater_{j}' for j in range(4)]:
        arr = df[col].to_numpy(float, copy=True)
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = np.nanmedian(np.where(np.isfinite(arr[is_ref]), arr[is_ref], np.nan))
            df[col] = arr
    z = pd.DataFrame(index=df.index)
    for f in FIELDS:
        x = df[f].to_numpy(float)
        ref = x[is_ref]
        if f.startswith('modernbert_') or f in ('ad_en', 'fluency_en'):
            v = np.clip(x, 0, 1)
        elif f == 'fineweb_edu':
            v = np.clip(x / 5, 0, 1)
        elif f == 'qurater':
            v = np.mean([ref_cdf(df.loc[is_ref, f'qurater_{j}'].to_numpy(), w, df[f'qurater_{j}'].to_numpy())
                         for j in range(4)], axis=0)
        elif f.startswith('dsir_'):                          # per-word importance, then percentile
            per_word = x / np.maximum(df['rps_doc_word_count'].to_numpy(), 1)
            v = ref_cdf(per_word[is_ref], w, per_word)
        elif f == 'rps_doc_mean_word_length':                # trapezoid: full score for 3 to 8 characters
            v = np.minimum(np.minimum((x - 1) / 2, (15 - x) / 7), 1).clip(0, 1)
        elif f in ('rps_doc_word_count', 'rps_doc_num_sentences'):
            lo, hi = weighted_quantile(np.log1p(ref.clip(0)), w, [0.05, 0.75])
            v = np.clip((np.log1p(x.clip(0)) - lo) / (hi - lo), 0, 1)
        else:
            v = ref_cdf(ref, w, x)
            if f in NEGATIVE:
                v = 1 - v
        z[f] = v
    G = np.column_stack([z[fs].mean(axis=1) for fs in GROUPS.values()])
    out = df[['domain', 'id']].copy()
    for j, g in enumerate(GROUPS):
        out[g] = G[:, j]
    out['Q'] = smoothed_geometric_mean(G)
    out['Q_arithmetic'] = G.mean(axis=1)
    out['Q_equal22'] = z.mean(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', nargs='+', required=True, help='JSONL(.xz) files of the reference sample')
    ap.add_argument('--check', nargs='*', default=[], help='JSONL(.xz) files scored with the unchanged scale')
    ap.add_argument('--domain-field', default='_source_domain')
    ap.add_argument('--out', default='.')
    a = ap.parse_args()
    rows = [dict(r, ref=True) for f in a.reference for r in read_records(f, a.domain_field)]
    rows += [dict(r, ref=False) for f in a.check for r in read_records(f, a.domain_field)]
    df = pd.DataFrame(rows)
    is_ref = df['ref'].to_numpy()
    out = score(df, is_ref)
    first = ~df.duplicated(['domain', 'id']).to_numpy()
    sets = {'reference': is_ref, 'check_new': (~is_ref) & first, 'all': first}
    summ = []
    for name, sel in sets.items():
        for dom, sub in out.loc[sel].groupby('domain'):
            summ.append({'set': name, 'domain': dom, 'n': len(sub), 'Q_mean': sub['Q'].mean(),
                         **{c: sub[c].mean() for c in list(GROUPS) + ['Q_arithmetic', 'Q_equal22']}})
    summ = pd.DataFrame(summ)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    summ.to_csv(Path(a.out) / 'quality_domain_summary.csv', index=False)
    q0 = summ.loc[summ['set'] == 'all', 'Q_mean'].mean()
    q0_ref = summ.loc[summ['set'] == 'reference', 'Q_mean'].mean()
    print(summ[['set', 'domain', 'n', 'Q_mean']].to_string(index=False))
    print(f'documents: reference {int(is_ref.sum())}, new check documents {int(sets["check_new"].sum())}, '
          f'distinct total {int(first.sum())}')
    print(f'baseline quality Q0 (equal-domain mean over all distinct documents) = {q0:.16f}')
    print(f'equal-domain mean of the reference sample alone = {q0_ref:.16f}')


if __name__ == '__main__':
    main()
