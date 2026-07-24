"""
filter_youtube_commons_v3.py  — strict-title build
Build a finance/crypto ("hustlesphere") corpus from PleIAs/YouTube-Commons.

v3 changes (driven by a hand-labelled validation sample where v2 scored 0.34 precision):
  * STRICT lexicon: bare "trading"/"stocks" REMOVED (they matched "stock footage",
    Pokémon "trading", soap-opera "insider trading"). Replaced with finance-only phrases
    ("stock market", "options trading", "crypto trading", "common stocks", ...) plus
    unambiguous crypto tokens. Validated at ~0.92 precision / 0.65 recall on the title.
  * TITLE-HIT IS THE DEFAULT keep rule. The old "N transcript hits" rule was the main
    precision killer (long lectures/audiobooks rack up incidental hits) and is now OFF by
    default. Re-enable a stricter version with --use-transcript if you want more recall.
  * --semantic second pass: sentence-transformer relevance score over titles, to strip the
    last few off-topic survivors (e.g. astrology "Investing Wisely"). Use with --semantic-threshold.

Everything else (per-shard polars engine, resumability, bounded disk, global dedupe,
manifest, stats, --emit-ids, --validate) is unchanged from v2.

NB: YouTube-Commons is CC-BY-only -> skews to lectures/audiobooks/conference talks, NOT
    retail finfluencer hype. Good for Phase-1 topic modelling; RQ2 (misleading claims) and
    engagement features still need the yt-dlp path.

    pip install polars pyarrow huggingface_hub
    pip install sentence-transformers          # only for --semantic
    python filter_youtube_commons_v3.py --limit-shards 2 --validate 50      # smoke + re-label
    python filter_youtube_commons_v3.py --semantic --semantic-threshold 0.30 --emit-ids
"""
import os
import csv
import gc
import json
import glob
import argparse
import datetime as dt
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

# ---- STRICT finance/crypto lexicon (validated ~0.92 precision on titles) ----------
TERMS = [
    # crypto — unambiguous single tokens
    r"crypto", r"cryptocurrenc(?:y|ies)", r"bitcoin", r"ethereum", r"altcoins?",
    r"blockchain", r"defi", r"web3", r"nft", r"hodl", r"memecoins?",
    r"dogecoin", r"solana", r"ripple", r"xrp", r"binance", r"coinbase",
    # trading / markets — require finance phrasing (NO bare "trading"/"stocks")
    r"forex", r"day ?trad(?:ing|er)", r"swing trading", r"options trading",
    r"crypto ?trading", r"stock trading", r"trading bot", r"trading strateg(?:y|ies)",
    r"stock market", r"stock options", r"penny stocks?", r"common stocks?",
    r"stock picks?", r"trading stocks?",
    # investing
    r"investing", r"investments?", r"how to invest", r"index funds?",
    # make-money / hustle cluster
    r"passive income", r"side hustle", r"side income", r"affiliate marketing",
    r"make money online", r"making money online", r"earn money online",
    r"how to make money", r"financial freedom", r"get rich", r"personal finance",
    r"dropshipping", r"e-?commerce",
]
PATTERN = r"(?i)\b(?:" + "|".join(TERMS) + r")\b"

PROTOTYPES = [
    "cryptocurrency trading and bitcoin investing",
    "how to make money online with a side hustle",
    "stock market investing and day trading",
    "dropshipping and e-commerce business",
    "passive income and financial freedom",
]

WANT_COLS = ["video_id", "channel_id", "channel", "title", "text", "date",
             "word_count", "original_language", "transcription_language"]


def present_columns(path):
    have = set(pq.read_schema(path).names)
    return [c for c in WANT_COLS if c in have]


def process_shard(path, cols, use_transcript, min_text_hits, min_density):
    df = pl.read_parquet(path, columns=cols)

    lang_ok = pl.lit(True)
    if "original_language" in cols:
        lang_ok = lang_ok & (pl.col("original_language") == "en")
    if "transcription_language" in cols:
        lang_ok = lang_ok & (pl.col("transcription_language") == "en")

    title_hit = pl.col("title").fill_null("").str.contains(PATTERN)
    text_hits = (pl.col("text").fill_null("").str.count_matches(PATTERN)
                 if "text" in cols else pl.lit(0))

    df = df.with_columns([title_hit.alias("title_hit"), text_hits.alias("text_hits")])

    # transcript cleaning + density (computed for analysis regardless)
    text_col = pl.col("text") if "text" in cols else pl.lit("")
    df = df.with_columns(
        text_col.fill_null("")
                .str.replace_all(r"\[[^\]]*\]", " ")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
                .alias("transcript")
    )
    df = df.with_columns(pl.col("transcript").str.split(" ").list.len().alias("n_words"))
    df = df.with_columns(
        (pl.col("text_hits") / (pl.col("n_words") + 1) * 1000).alias("term_density")
    )

    # KEEP RULE: title hit by default; optionally OR-in strict transcript evidence
    keep = pl.col("title_hit")
    if use_transcript:
        keep = keep | ((pl.col("text_hits") >= min_text_hits) &
                       (pl.col("term_density") >= min_density))
    df = df.filter(lang_ok & keep)
    if df.height == 0:
        return df

    return df.select([
        pl.col("video_id").alias("display_id"),
        (pl.col("channel_id") if "channel_id" in cols else pl.lit(None)).alias("channel_id"),
        (pl.col("channel") if "channel" in cols else pl.lit(None)).alias("channel"),
        pl.col("title"),
        (pl.col("date").cast(pl.Utf8).str.slice(0, 10) if "date" in cols
         else pl.lit(None)).alias("upload_date"),
        pl.col("transcript"),
        pl.col("n_words").alias("word_count"),
        pl.col("title_hit"),
        pl.col("text_hits"),
        pl.col("term_density"),
        pl.lit("en").alias("language"),
        pl.lit("youtube_commons").alias("source"),
    ])


def semantic_rerank(final_path, threshold):
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")
    proto = model.encode(PROTOTYPES, convert_to_tensor=True, normalize_embeddings=True)
    df = pl.read_parquet(final_path)
    emb = model.encode(df["title"].fill_null("").to_list(), batch_size=256,
                       convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
    rel = util.cos_sim(emb, proto).max(dim=1).values.cpu().numpy()
    df = df.with_columns(pl.Series("relevance", rel))
    if threshold is not None:
        before = df.height
        df = df.filter(pl.col("relevance") >= threshold)
        print(f"semantic filter: {before} -> {df.height} at relevance >= {threshold}")
    df.write_parquet(final_path)


def write_stats(final_path, out_dir):
    df = pl.read_parquet(final_path)
    by_year = (df.with_columns(pl.col("upload_date").str.slice(0, 4).alias("yr"))
                 .group_by("yr").len().sort("yr"))
    by_year.write_csv(out_dir / "stats_by_year.csv")
    (df.group_by("channel").len().sort("len", descending=True).head(25)
       .write_csv(out_dir / "stats_top_channels.csv"))
    print(f"\nkept {df.height:,} videos | {df['channel_id'].n_unique():,} channels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PleIAs/YouTube-Commons")
    ap.add_argument("--out-dir", default="hustlesphere_ytc")
    ap.add_argument("--use-transcript", action="store_true",
                    help="also admit videos on strict transcript evidence (more recall, less precision)")
    ap.add_argument("--min-text-hits", type=int, default=4)
    ap.add_argument("--min-density", type=float, default=0.5, help="strict-term hits per 1k words")
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--keep-shards", action="store_true")
    ap.add_argument("--semantic", action="store_true")
    ap.add_argument("--semantic-threshold", type=float, default=None)
    ap.add_argument("--validate", type=int, default=0)
    ap.add_argument("--emit-ids", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    parts_dir = out_dir / "parts"; parts_dir.mkdir(exist_ok=True)
    shard_dir = out_dir / "_shards"; shard_dir.mkdir(exist_ok=True)
    ckpt = out_dir / "checkpoint.json"
    done = set(json.loads(ckpt.read_text())) if ckpt.exists() else set()

    files = sorted(f for f in HfApi().list_repo_files(args.dataset, repo_type="dataset")
                   if f.endswith(".parquet"))
    if args.limit_shards:
        files = files[:args.limit_shards]
    print(f"{len(files)} parquet shards; {len(done)} already done")

    for i, f in enumerate(files):
        if f in done:
            continue
        local = hf_hub_download(args.dataset, f, repo_type="dataset", local_dir=str(shard_dir))
        try:
            cols = present_columns(local)
            kept = process_shard(local, cols, args.use_transcript,
                                 args.min_text_hits, args.min_density)
            if kept.height:
                kept.write_parquet(parts_dir / f"part-{i:05d}.parquet")
            print(f"[{i+1}/{len(files)}] {f}: kept {kept.height}")
        finally:
            if not args.keep_shards:
                try: os.remove(local)
                except OSError: pass
            gc.collect()
        done.add(f); ckpt.write_text(json.dumps(sorted(done)))

    final = out_dir / "hustlesphere_ytc.parquet"
    parts = glob.glob(str(parts_dir / "*.parquet"))
    if not parts:
        print("no rows matched."); return
    pl.scan_parquet(parts).unique(subset="display_id", keep="first").sink_parquet(final)
    print(f"\nmerged -> {final}")

    if args.semantic:
        semantic_rerank(final, args.semantic_threshold)
    write_stats(final, out_dir)

    if args.emit_ids:
        pl.read_parquet(final).select("display_id", "channel_id").write_csv(
            out_dir / "kept_video_ids.csv")
        print("kept_video_ids.csv written — feed to yt-dlp to backfill view/like/duration")

    if args.validate:
        df = pl.read_parquet(final)
        df = df.sample(min(args.validate, df.height))
        with open(out_dir / "validation_sample.csv", "w", newline="", encoding="utf-8") as cf:
            w = csv.writer(cf); w.writerow(["display_id", "title", "snippet", "relevance", "label"])
            for r in df.iter_rows(named=True):
                w.writerow([r["display_id"], r["title"], (r["transcript"] or "")[:300],
                            r.get("relevance", ""), ""])
        print("validation_sample.csv — re-label 'label' (1/0); precision = mean(label)")

    (out_dir / "manifest.json").write_text(json.dumps({
        "dataset": args.dataset, "version": "v3-strict-title", "lexicon": TERMS,
        "use_transcript": args.use_transcript, "min_text_hits": args.min_text_hits,
        "min_density": args.min_density, "semantic": args.semantic,
        "semantic_threshold": args.semantic_threshold, "shards_processed": len(done),
        "polars": pl.__version__, "run_finished": dt.datetime.now(dt.timezone.utc).isoformat(),
    }, indent=2))


if __name__ == "__main__":
    main()
