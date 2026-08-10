#!/usr/bin/env python3
"""Hard acceptance gate for NLP From Scratch S00–S16."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"

def load(name: str):
    return json.loads((RES / f"{name}.json").read_text())

def main() -> int:
    fails = []
    def need(cond, msg):
        if not cond:
            fails.append(msg)

    d = load("S00_bow_naive_bayes"); need(d["metric"]["test_acc"] >= 0.8, "S00")
    d = load("S01_tfidf_logreg"); need(d["metric"]["test_acc"] >= 0.8, "S01")
    tops = [t for t, _ in d["top_features"].get("pos", [])]
    need(all(t not in {"the", "and", "of", "in"} for t in tops[:3]), f"S01 stopword tops {tops}")

    d = load("S02_embeddings_similarity")
    rel = next(p for p in d["sentence_pairs"] if "Plants" in p["a"] or "plants" in p["a"])
    unrel = next(p for p in d["sentence_pairs"] if "team" in p["b"].lower())
    need(rel["emb_cos"] > unrel["emb_cos"], "S02 ordering")

    d = load("S03_bm25_ranking")
    need(d["rankings"]["how do plants make food from sunlight"][0]["title"] == "Photosynthesis", "S03")

    d = load("S04_textrank_summarize")
    need(len(d["results"]) >= 4, "S04")

    d = load("S05_retrieval_qa"); need(d["metric"]["soft_accuracy"] >= 0.75, "S05")
    for fq in d.get("free_queries", []):
        if fq["question"].startswith("What absorbs"):
            need("chlorophyll" in fq["answer"].lower(), f"S05 free {fq['answer']}")

    d = load("S06_neural_cls"); need(d["metric"]["test_acc"] >= 0.75, "S06")
    need(d.get("device_count", 0) >= 2, f"S06 dual GPU {d.get('device_count')}")
    d = load("S07_bilstm_ner"); need(d["metric"]["heldout_exact_seq"] >= 0.5, "S07")
    d = load("S08_rnn_lm"); need(bool(d.get("demos")), "S08")
    d = load("S09_seq2seq_mt"); need(d["metric"]["token_recall"] >= 0.5, "S09")

    d = load("S10_tiny_transformer")
    feats = d["features"]
    if all("related" in f for f in feats):
        rels = [f["cos"] for f in feats if f["related"]]
        unre = [f["cos"] for f in feats if not f["related"]]
        need(min(rels) > max(unre), f"S10 feat order rel={rels} unrel={unre}")
    else:
        need(feats[0]["cos"] > feats[1]["cos"], "S10 feat order")
    if "min_related" in d.get("metric", {}):
        need(d["metric"]["min_related"] > d["metric"]["max_unrelated"], "S10 metric ordering")

    d = load("S11_pretrained_encoder")
    need("paris" in d["fill_mask"]["top"][0]["token"].lower(), "S11 fill token")
    need(d["similarity"][0]["cos"] > d["similarity"][1]["cos"], "S11 sim order")
    qa0 = d["qa"][0]["answer"].lower()
    need(any(x in qa0 for x in ("oxygen", "glucose")), f"S11 plants QA {qa0!r}")


    d = load("S12_t5_text2text")
    need(bool(d["translation"]["output"]) and bool(d["summarization"]["output"]), "S12")

    d = load("S13_zero_shot")
    need(d["demos"][0]["labels"][0] == "sports", f"S13 {d['demos'][0]['labels'][:2]}")
    sci = next((x for x in d["demos"] if "chlorophyll" in x["text"].lower() or x.get("gold")=="science"), None)
    if sci is not None:
        need(sci["labels"][0] == "science" or sci.get("gold") == sci["labels"][0], f"S13 science→{sci['labels'][0]}")


    d = load("S14_cross_encoder_rank")
    need(d["demos"][0]["ranking"][0]["id"] == "d1", "S14 rank")

    d = load("S15_table_qa")
    need(
        d.get("method")
        in {"tapas", "neural_row_retrieve_plus_column_project", "neural_cell_scorer_fallback"}
        or d.get("tapas_available"),
        f"S15 method {d.get('method')}",
    )
    hits = d.get("hits") or {}
    need(hits.get("pandas", 0) == 4 or all(x.get("ok_pandas") for x in d["demos"]), f"S15 pandas {hits}")
    need(hits.get("neural", 0) >= 3 or hits.get("tapas", 0) >= 3, f"S15 neural/tapas {hits}")

    d = load("S16_unified_pipeline")
    need(all(d["acceptance"].values()), f"S16 {d['acceptance']}")
    tok = str(d["unified"].get("fill_mask_token") or "")
    need("paris" in tok.lower(), f"S16 fill token {tok!r}")

    if fails:
        print("FAIL", fails)
        return 1
    print("ALL ACCEPTANCE CHECKS PASSED (S00–S16)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
