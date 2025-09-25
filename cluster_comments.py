import os, argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if "text" not in df.columns:
        raise SystemExit("CSV must contain a 'text' column.")
    if "like_count" in df.columns:
        df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype(int)
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)
    return df

def embed(texts, model_name="all-MiniLM-L6-v2", batch_size=64):
    model = SentenceTransformer(model_name)
    vec = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(vec, dtype=np.float32)

def cluster_dbscan(emb, sim=0.88, min_samples=3):
    eps = 1.0 - float(sim)  # cosine distance
    db = DBSCAN(metric="cosine", eps=eps, min_samples=min_samples, n_jobs=-1)
    return db.fit_predict(emb)  # -1 = noise

def summarize(df_with_ids: pd.DataFrame) -> pd.DataFrame:
    # add a real column once so sort_values can reference it by name
    df = df_with_ids.copy()
    if "like_count" in df.columns:
        df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype(int)
    df["text_len"] = df["text"].astype(str).str.len()

    rows = []
    for cid, g in df.groupby("cluster_id"):
        # representative: highest-like; tie-breaker: longest
        rep = g.sort_values(by=["like_count", "text_len"], ascending=False).iloc[0]
        rows.append({
            "cluster_id": int(cid),
            "size": int(len(g)),
            "top_likes": int(rep.get("like_count", 0)),
            "representative": rep["text"][:300] + ("…" if len(rep["text"]) > 300 else "")
        })
    return pd.DataFrame(rows).sort_values(by="size", ascending=False)


def main():
    ap = argparse.ArgumentParser(description="Semantic clustering of comments from CSV.")
    ap.add_argument("csv", help="Input CSV from fetch step")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--sim", type=float, default=0.88, help="Higher = tighter (0..1)")
    ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--csv-base", default=None, help="Base name for outputs; defaults to input name without .csv")
    args = ap.parse_args()

    df = load_df(args.csv)
    if df.empty:
        print("No comments to cluster."); return

    print(f"Encoding {len(df)} comments with {args.model} …")
    emb = embed(df["text"].tolist(), model_name=args.model)

    print(f"Clustering with DBSCAN (sim≥{args.sim:.2f}, min_samples={args.min_samples}) …")
    labels = cluster_dbscan(emb, sim=args.sim, min_samples=args.min_samples)

    base = args.csv_base or os.path.splitext(args.csv)[0]
    clustered_path = f"{base}_clustered.csv"
    summary_path = f"{base}_clusters_summary.csv"

    out = df.copy()
    # Re-map labels to 1..K (keep noise as its own singletons with unique ids)
    # Build groups
    groups = {}
    next_id = 1
    for i, lab in enumerate(labels):
        if lab == -1:
            groups[next_id] = [i]; next_id += 1
        else:
            groups.setdefault(lab, []).append(i)
    # Sort by size desc, then assign compact ids
    ordered = sorted(groups.values(), key=len, reverse=True)
    id_map = {}
    cid = 1
    for g in ordered:
        for idx in g: id_map[idx] = cid
        cid += 1

    out["cluster_id"] = [id_map[i] for i in range(len(out))]
    out.to_csv(clustered_path, index=False, encoding="utf-8")

    summary = summarize(out)
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    total = out["cluster_id"].nunique()
    sizes = out.groupby("cluster_id").size().sort_values(ascending=False)
    singletons = int((sizes == 1).sum())
    multis = int(total - singletons)

    print(f"Done. Wrote:\n  - {clustered_path}\n  - {summary_path}")
    print(f"Clusters: {total} total  |  ≥2 size: {multis}  |  singletons: {singletons}")
    print("\nTop 5 clusters:")
    for cid, sz in sizes.head(5).items():
        rep = summary[summary["cluster_id"] == cid]["representative"].values[0]
        print(f"  [{int(cid)}] size={int(sz)}  rep: {rep[:100]}{'…' if len(rep)>100 else ''}")

if __name__ == "__main__":
    main()
