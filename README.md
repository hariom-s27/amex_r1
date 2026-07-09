<div align="center">

# 💳 Amex Campus Challenge 2026 — Round 1
### Cardmember Profitability Ranking

**Rank 500,000 Premier Card members by estimated profitability. Identify the top 20%.**

[![Score](https://img.shields.io/badge/Public_Score-0.923-2ea44f?style=for-the-badge)](.)
[![Climb](https://img.shields.io/badge/Climb-0.875→0.923-blue?style=for-the-badge)](.)
[![Stability](https://img.shields.io/badge/Fold_Stability-99.9%25-orange?style=for-the-badge)](.)

</div>

---

## 📌 The Problem

American Express provided **500,000 masked cardmember records** (23 anonymized attributes, no PII) and one task:

> Design a **profitability equation** to rank-order members and identify the **top 20% most profitable**.

- **No target label.** The true profitability is hidden.
- **Metric:** accuracy = overlap between our predicted top-20% and the actual top-20%.
- **Split:** 70% public leaderboard (live), 30% private leaderboard (hidden until round close).
- **Constraint:** an interpretable equation using existing variables — no identifiers, no synthetic rows.

---

## 🏆 Result

| Metric | Value |
|---|---|
| **Final public score** | **0.923** |
| Starting baseline | 0.875 |
| Total gain | **+4.8 points** |
| 5-fold top-set stability | 99.9% (private-board safe) |

---

## 🧗 The Climb — every gain backed by a real leaderboard score

| Step | Score | Lever discovered |
|---|:---:|---|
| Baseline P&L | 0.875 | Balanced spend + revolve − benefits − risk |
| **Exclude unobservable members** | 0.894 | **+1.0** — members with no observed spend can't be ranked → rank them low |
| **Weight revolving interest up** | 0.921 | **+2.4** — interest (≈20% APR) ≫ interchange (≈1%); the biggest lever |
| **Tune risk severity** | 0.922 | +0.1 — heavier revolve made risk newly relevant (lgd 0.85 → 1.5) |
| **Calibrate benefit costs** | **0.923** | +0.1 — corrected cab-credit unit weight |

Each step was one isolated change, uploaded, and confirmed on the board before the next.

---

## 🧠 The Final Model

```
Profit_i = 0.025 · Spend_i                      ← net interchange on spend
         + 0.35  · f1_i                          ← net interest on revolving balance
         − (35·f13 + f14 + 40·f15 + f16)         ← benefit usage priced at unit cost
         − 1.5 · f11_i · (f1_i + Spend_i/12)     ← expected credit loss (risk × exposure)

  Spend_i = f6 + f7 + … + f10   (category spends, floored at 0)
  Members with no observed category spend are ranked below all others.
```

**Business reading:** engaged, low-risk revolvers with real spend rank highest; distressed high-risk revolvers and members we cannot observe rank lowest — exactly how an issuer values a portfolio.

---

## 🔬 Method — how we found the levers without labels

With no target, the leaderboard *is* the validation signal. The workflow:

1. **Isolate one lever** per experiment (change exactly one thing).
2. **Audit offline** — a mandatory pre-submission gate (`src/audit_submission.py`) checks structure, IDs, ties, framework text, geometry vs. all prior scores, and a 5-fold stability simulation.
3. **Upload, read the real score**, add it as an anchor.
4. **Keep only evidence-backed gains** — every dead end (f5, benefits-off, segment ranking, engagement, category re-weighting) was killed by a real score, never a guess.

Key ideas that worked: **set-overlap math** (unobservable members carry only the 20% base rate → rank them low), and **credit-card economics** (revolving interest dominates profit → weight it heavily).

---

## 📁 Repository Structure

```
amex_r1/
├── README.md                  ← you are here
├── src/                       ← the reusable engine
│   ├── config.py              ← all paths & constants (single source)
│   ├── data_io.py             ← cached loading, frozen imputer
│   ├── scoring.py             ← the profit equation: build(), top_set(), overlap()
│   ├── submit.py              ← writer + framework text + stale-text guard
│   ├── audit_submission.py    ← the pre-submission gate (run before every upload)
│   └── anchors.json           ← REAL leaderboard scores only
├── experiments/               ← one script per hypothesis (the climb)
├── outputs/submissions/       ← scored submission files
├── docs/
│   ├── JOURNEY.md             ← the full experiment log, every score explained
│   └── framework.md           ← the profitability framework write-up
└── data/raw/                  ← dataset & template (not committed)
```

---

## 🚀 Reproduce

```bash
pip install pandas numpy scikit-learn openpyxl pyarrow
# place the dataset in data/raw/
python experiments/build_champion.py               # builds the 0.923 model
python -m src.audit_submission outputs/submissions/champion.xlsx   # verify
```

---

## 📖 What This Project Demonstrates

- Reverse-engineering a hidden objective from **leaderboard feedback alone**.
- **Disciplined experimentation** — one lever at a time, every result audited, no leaderboard-chasing.
- Translating **domain economics** (issuer P&L) into a simple, interpretable, defensible model.
- A **+4.8-point climb** from a strong baseline to a private-board-stable final score.

<div align="center">

*Built for the American Express Campus Challenge 2026.*

</div>
