#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Niels Henrik Abel (1802-1829) - norveški matematičar
algebra and mathematical analysis
group theory and elliptic functions
"""

"""
abel_3_v1_grupa.py — Abelova grupa → next Loto / Loto Plus

Grupa G ≅ (Z/39Z, +) na brojevima {1..39}:
  a ⊕ b = ((a-1 + b-1) mod 39) + 1
  e = 1
  inv(a) = ((1 - (a-1)) mod 39) + 1

fold(C) = c1 ⊕ c2 ⊕ … ⊕ c7   (komutativno, asocijativno)

Tok:
  1. Niz g[t] = fold(draws[t])
  2. g_hat = argmin_g Abel-tezinska kružna udaljenost do istorije g
       cost(g) = Σ_j r^j · circ(g, g[n-1-j]),  r = 39/40
  3. skor broja k: Abel-tezinska frekvencija u istoriji
  4. next = validna kombinacija sa fold(C)=g_hat i max Σ skor(c),
       tie-break SEED

Bez RNG. CSV: loto_2949 i loto_plus_1705.
"""

from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import numpy as np

SEED = 39
N_PICK = 7
MAX_NUM = 39
MIN_HIST = 100
ABEL_R = float(SEED) / float(SEED + 1)  # 39/40

ROOT = Path(__file__).resolve().parent
CSV_LOTO = ROOT.parent / "data" / "loto7_4654_k58_loto_2949.csv"
CSV_PLUS = ROOT.parent / "data" / "loto7_4654_k58_loto_plus_1705.csv"
CSV_PATH = CSV_LOTO

POS_LO = np.arange(1, N_PICK + 1, dtype=int)
POS_HI = POS_LO + (MAX_NUM - N_PICK)


def load_draws(path: Path = CSV_PATH) -> np.ndarray:
    raw = np.loadtxt(path, delimiter=",", dtype=int)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    assert raw.shape[1] == N_PICK, raw.shape
    assert raw.min() >= 1 and raw.max() <= MAX_NUM
    return raw


def op(a: int, b: int) -> int:
    """Abelova operacija ⊕ na {1..MAX_NUM} ≅ Z/MAX_NUM Z."""
    return ((a - 1 + b - 1) % MAX_NUM) + 1


def fold(combo: Tuple[int, ...] | List[int] | np.ndarray) -> int:
    g = int(combo[0])
    for i in range(1, len(combo)):
        g = op(g, int(combo[i]))
    return g


def circ_dist(a: int, b: int) -> int:
    """Kružna udaljenost na grupi (residues 0..38)."""
    da = (a - 1) % MAX_NUM
    db = (b - 1) % MAX_NUM
    d = abs(da - db)
    return int(min(d, MAX_NUM - d))


def series_g(draws: np.ndarray) -> np.ndarray:
    n = len(draws)
    out = np.empty(n, dtype=int)
    for t in range(n):
        out[t] = fold(draws[t])
    return out


def predict_g_hat(g: np.ndarray) -> int:
    """Abel-tezinski najbliži grupni element."""
    n = len(g)
    r = ABEL_R
    best_g = 1
    best_key: Tuple[float, int, int] | None = None
    for cand in range(1, MAX_NUM + 1):
        cost = 0.0
        rj = 1.0
        for j in range(n):
            cost += rj * float(circ_dist(cand, int(g[n - 1 - j])))
            rj *= r
        seed_tie = (cand * SEED) % 97
        key = (cost, seed_tie, cand)
        if best_key is None or key < best_key:
            best_key = key
            best_g = cand
    return best_g


def number_scores(draws: np.ndarray) -> np.ndarray:
    """Abel-tezinska frekvencija po broju 1..MAX_NUM (index 0 unused)."""
    n = len(draws)
    r = ABEL_R
    score = np.zeros(MAX_NUM + 1, dtype=np.float64)
    rj = 1.0
    for j in range(n):
        row = draws[n - 1 - j]
        for x in row:
            score[int(x)] += rj
        rj *= r
    return score


def valid_positions(combo: Tuple[int, ...]) -> bool:
    return all(int(POS_LO[i]) <= combo[i] <= int(POS_HI[i]) for i in range(N_PICK))


def invert_g_to_combo(
    g_hat: int, scores: np.ndarray
) -> Tuple[List[int], float]:
    """Među fold(C)=g_hat, max skor; tie SEED."""
    best_combo: Tuple[int, ...] | None = None
    best_score = -1.0
    best_key: Tuple[float, int, Tuple[int, ...]] | None = None
    for comb in combinations(range(1, MAX_NUM + 1), N_PICK):
        if not valid_positions(comb):
            continue
        if fold(comb) != g_hat:
            continue
        s = float(sum(scores[c] for c in comb))
        # maksimizuj skor ⇒ ključ sa -s
        seed_tie = (sum(comb) * SEED) % 97
        key = (-s, seed_tie, comb)
        if best_key is None or key < best_key:
            best_key = key
            best_combo = comb
            best_score = s
    if best_combo is None:
        raise RuntimeError(f"nema kandidata za g_hat={g_hat}")
    return list(best_combo), best_score


def predict_next(draws: np.ndarray) -> dict:
    g = series_g(draws)
    assert len(g) == len(draws)
    g_hat = predict_g_hat(g)
    scores = number_scores(draws)
    nxt, score_match = invert_g_to_combo(g_hat, scores)
    g_match = fold(nxt)
    return {
        "g": g,
        "g_last": int(g[-1]),
        "g_hat": int(g_hat),
        "g_match": int(g_match),
        "method": "abel_group",
        "abel_r": ABEL_R,
        "next": nxt,
        "score_match": float(score_match),
        "last_combo": draws[-1].tolist(),
    }


def _print_one(label: str, csv_path: Path, r: dict) -> None:
    _, counts = np.unique(r["g"], return_counts=True)
    n_dup = int(np.sum(counts > 1))
    print(f"=== {label} ===")
    print("csv:             ", csv_path.name)
    print("csv rows:        ", len(r["g"]))
    print("last combo:      ", r["last_combo"])
    print("g_last:          ", r["g_last"])
    print("method:          ", r["method"])
    print("g_hat:           ", r["g_hat"])
    print("g_match:         ", r["g_match"])
    print("score_match:     ", format(r["score_match"], ".12f"))
    print("g hist dup:      ", n_dup)
    print(f"next_{label}:     ", r["next"])


def main() -> None:
    draws_loto = load_draws(CSV_LOTO)
    draws_plus = load_draws(CSV_PLUS)
    assert len(draws_loto) >= MIN_HIST + 2
    assert len(draws_plus) >= MIN_HIST + 2

    r_loto = predict_next(draws_loto)
    r_plus = predict_next(draws_plus)

    _print_one("loto", CSV_LOTO, r_loto)
    print()
    _print_one("loto_plus", CSV_PLUS, r_plus)
    print()
    print("next_loto:      ", r_loto["next"])
    print("next_loto_plus: ", r_plus["next"])


if __name__ == "__main__":
    main()



########################################################



"""
RUN:

=== loto ===
csv:              loto7_4654_k58_loto_2949.csv
csv rows:         2949
last combo:       [6, 8, 12, 22, 30, 36, 38]
g_last:           29
method:           abel_group
g_hat:            5
g_match:          5
score_match:      66.604674039542
g hist dup:       39
next_loto:      [6, 7, 14, 20, 22, 29, 30]

=== loto_plus ===
csv:              loto7_4654_k58_loto_plus_1705.csv
csv rows:         1705
last combo:       [3, 6, 9, 13, 18, 20, 21]
g_last:           6
method:           abel_group
g_hat:            22
g_match:          22
score_match:      69.072150394053
g hist dup:       39
next_loto_plus:      [1, 3, 8, 11, 18, 31, 34]

next_loto:       [6, 7, 14, 20, 22, 29, 30]
next_loto_plus:  [1, 3, 8, 11, 18, 31, 34]
"""



########################################################################################



"""
TEST:


Backtest Abel grupa (n−1):


Loto (2948 → actual 2949)

pred: [1, 6, 14, 20, 28, 29, 30]
actual: [6, 8, 12, 22, 30, 36, 38]
HIT: False
· 2/7 (6, 30)
g_hat: 5
g_match: 5
score_match: 66.049221893678


Loto Plus (1704 → actual 1705)

pred: [1, 4, 7, 11, 18, 31, 34]
actual: [3, 6, 9, 13, 18, 20, 21]
HIT: False
· 1/7 (18)
g_hat: 22
g_match: 22
score_match: 69.084131428881



Backtest Abel grupa (n−2):


Loto (2947 → actual 2948)

pred: [1, 14, 16, 20, 22, 28, 29]
actual: [7, 15, 20, 26, 28, 30, 39]
HIT: False
· 2/7 (20, 28)
g_hat: 7
g_match: 7
score_match: 65.990125811271


Loto Plus (1703 → actual 1704)

pred: [1, 4, 7, 11, 18, 31, 34]
actual: [7, 8, 14, 15, 17, 23, 32]
HIT: False
· 1/7 (7)
g_hat: 22
g_match: 22
score_match: 69.829878388595



Backtest Abel grupa (n−3):


Loto (2946 → actual 2947)

pred: [1, 5, 6, 14, 16, 22, 28]
actual: [1, 4, 7, 20, 27, 29, 39]
HIT: False
· 1/7 (1)
g_hat: 8
g_match: 8
score_match: 65.919410924506


Loto Plus (1702 → actual 1703)

pred: [1, 3, 8, 11, 18, 31, 34]
actual: [4, 5, 6, 11, 12, 18, 28]
HIT: False
· 2/7 (11, 18)
g_hat: 22
g_match: 22
score_match: 69.261747925949
"""



########################################################################################



"""
ANALIZA — abel_3_v1_grupa.py:

1. Ulaz — dva čista CSV-a: Loto (2949) i Plus (1705), bez miksa.

2. Abelova grupa G ≅ (Z/39Z, +) na {1..39}:
   a ⊕ b = ((a-1 + b-1) mod 39) + 1
   e = 1
   fold(C) = c1 ⊕ … ⊕ c7

3. Niz g — g[t] = fold(draws[t]).

4. Forecast g_hat — Abel-tezinska kružna udaljenost, r = 39/40:
   cost(g) = Σ_j r^j · circ(g, g[n-1-j])
   g_hat = argmin cost (tie-break SEED).

5. Skor broja — Abel-tezinska frekvencija u istoriji.
   next = validna C sa fold(C)=g_hat i max Σ skor(c), SEED tie-break.

6. Izlaz — next_loto i next_loto_plus.

Bitno: g_match = g_hat (grupni uslov drži); HIT False meri izbor kombinacije.

Backtest: n−1 Loto 2/7, Plus 1/7; n−2 Loto 2/7, Plus 1/7; n−3 Loto 1/7, Plus 2/7.
"""



########################################################################################



"""
BELESKE:

grupa (Z/39Z, ⊕) · fold · Abel g_hat · skor brojeva → next




Redosled rada (Abel → loto), jači prvi:
  1. Abelova suma / transformacija na nizu S → S_next_hat → invert  (ovo)
  2. Eliptičke / Abelove funkcije — nova mapa kombinacije → skalar, pa forecast + invert
  3. Abelova grupa — komutativna operacija na brojevima/parovima → skor → next
  (Abel-Ruffini / kvintika — ne u ovom redu.)
"""



########################################################################################