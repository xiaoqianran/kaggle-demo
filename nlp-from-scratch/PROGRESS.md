# PROGRESS — NLP From Scratch

| Stage | Status | Result | Where |
|-------|--------|--------|-------|
| S00 BoW+NB | ✅ | test_acc=1.00 | local |
| S01 TF-IDF+LogReg | ✅ | test_acc=1.00; stopword-filtered tops | local |
| S02 Embeddings | ✅ | related ≫ unrelated | local |
| S03 BM25 | ✅ | top-1 all queries | local |
| S04 TextRank | ✅ | extractive summaries | local |
| S05 Retrieval QA | ✅ | soft_acc=1.00; freeQA chlorophyll | local |
| S06 Neural CLS | ✅ | test_acc=1.0, **2×T4** | Kaggle neural |
| S07 BiLSTM NER | ✅ | exact-seq=1.0 | Kaggle neural |
| S08 RNN LM | ✅ | char generation demos | Kaggle neural |
| S09 Seq2Seq MT | ✅ | token_recall=1.0 | Kaggle neural |
| S10 Tiny Transformer | ✅ | min_rel≈0.999 > max_unrel≈0.088 | Kaggle neural v3 |
| S11 Pretrained Encoder | ✅ | Paris fill-mask; sim order OK | Kaggle modern |
| S12 T5 text2text | ✅ | MT/summary/gen-QA | Kaggle modern |
| S13 Zero-shot | ✅ | sports/business/movie | Kaggle modern |
| S14 Cross-Encoder | ✅ | plants→d1 | Kaggle modern |
| S15 Table QA | ✅ | pandas 4/4, neural row-retrieve ≥3/4 | Kaggle modern v5 |
| S16 Unified Pipeline | ✅ | **all acceptance True**; fill=paris | Kaggle modern v5 |

## Bugfix log (this pass)
- S01: filter stopwords from top features (`and`/`the`/`in`)
- S05: fix absorb-question span → `Chlorophyll in leaves`
- S10: supervised contrastive on labeled sentences; hard check min_rel>max_unrel
- S15: TAPAS token_type_ids regression on current transformers → neural row retrieval + column projection
- S16: use BERT `[MASK]` template `The capital of France is [MASK].` (assert token=paris)
- Removed empty scaffold notebook dirs

**Acceptance command:** `python nlp-from-scratch/scripts_accept.py`

Last update: bugfix + full acceptance PASS
