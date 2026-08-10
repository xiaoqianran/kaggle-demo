# PROGRESS — NLP From Scratch

| Stage | Status | Result | Where |
|-------|--------|--------|-------|
| S00–S05 | ✅ | classical IR/stats all pass | local |
| S06–S10 | ✅ | neural from-scratch on **2×T4** | Kaggle neural |
| S11 | ✅ | fill=paris; QA plants→oxygen; sim order OK | Kaggle modern v6 |
| S12 | ✅ | T5 MT/summary/gen-QA | Kaggle modern |
| S13 | ✅ | zero-shot sports/business/movie/**science** | Kaggle modern v6 |
| S14 | ✅ | CE rank plants→d1 | Kaggle modern |
| S15 | ✅ | pandas 4/4, neural row-QA ≥3/4 | Kaggle modern |
| S16 | ✅ | all acceptance True; fill token=paris | Kaggle modern |

**Gate:** `python nlp-from-scratch/scripts_accept.py` → ALL PASSED

## Latest bugfixes
- S11 QA context: explicit “produce oxygen and glucose”
- S13: DeBERTa zeroshot model; science≠sports
- S10 contrastive features ordered correctly
- S15 TAPAS wiring fallback → neural row retrieve
- S16 BERT [MASK] capital of France → paris

Last update: quality audit PASS (S11 QA + S13 science)
