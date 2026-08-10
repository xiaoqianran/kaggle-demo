# PROGRESS — NLP From Scratch

| Stage | Status | Result | Where |
|-------|--------|--------|-------|
| S00 BoW+NB | ✅ | test_acc=1.00 | local CPU (pure python) |
| S01 TF-IDF+LogReg | ✅ | test_acc=1.00 + feature importances | local |
| S02 Embeddings | ✅ | plants↔photo cos=0.58 ≫ movie↔sports 0.01 | local |
| S03 BM25 | ✅ | correct top-1 on all 5 queries | local |
| S04 TextRank | ✅ | extractive summaries vs lead-2 | local |
| S05 Retrieval QA | ✅ | soft_acc=1.00 (4/4) | local |
| S06 Neural CLS | ✅ | test_acc=1.0, 2×T4 DP | [Kaggle neural](https://www.kaggle.com/code/zhengyingxiong/grok-nlp-neural-from-scratch) |
| S07 BiLSTM NER | ✅ | heldout exact-seq=1.0 | Kaggle neural |
| S08 RNN LM | ✅ | char LM gen demos | Kaggle neural |
| S09 Seq2Seq MT | ✅ | token_recall=1.0 en→fr mini | Kaggle neural |
| S10 Tiny Transformer | ✅ | fill-mask + features + gen | Kaggle neural |
| S11 Pretrained Encoder | ✅ | fill-mask/CLS/NER/QA/sim | [Kaggle modern](https://www.kaggle.com/code/zhengyingxiong/grok-nlp-modern-frontier) |
| S12 T5 text2text | ✅ | MT/summary/gen-QA | Kaggle modern |
| S13 Zero-shot | ✅ | NLI zero-shot labels | Kaggle modern |
| S14 Cross-Encoder | ✅ | CE ≫ bi-encoder ranking | Kaggle modern |
| S15 Table QA | ✅ | TAPAS + pandas Tokyo/French/… | Kaggle modern |
| S16 Unified Pipeline | ✅ | **all 11 acceptance checks True** on T4×2 | Kaggle modern |

## Capability ladder (what you can *see*)

1. **S00–S01** Count words → classify docs; TF-IDF weights rare signal words  
2. **S02** Dense vectors → soft semantic similarity without exact word overlap  
3. **S03–S05** Rank docs, extract summaries, answer questions from a corpus  
4. **S06–S07** Neural end-to-end classification + per-token NER tags  
5. **S08–S09** Generate text; condition generation for translation  
6. **S10** Self-attention unifies fill-mask / features / generation  
7. **S11–S14** Pretraining + transfer: production quality multi-task + zero-shot + re-rank  
8. **S15–S16** Tables + full multi-task acceptance suite  

Last update: **route complete** (S00–S16)
