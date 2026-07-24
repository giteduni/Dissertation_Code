#!/usr/bin/env python3
"""
fastopic_best.py — high-quality hustlesphere taxonomy (title-only), instability-robust.

Strategy for "all core topics, no irrational fusions":
  * K=10 by default — enough budget to separate entrepreneurship from real estate,
    affiliate from e-commerce, freelance from generic WFH (which K=6 fused).
  * Wider vocabulary (min_df=5, vocab_size=30000) so higher K has lexical room and
    does NOT collapse into degenerate 'rich nomad' filler.
  * MULTI-SEED: fits N_SEEDS independent models and AUTOMATICALLY KEEPS THE BEST one
    by a combined score = mean_NPMI(non-degenerate topics) with a hard penalty for
    collapsed/duplicate topics. This defeats the run-to-run instability that lost
    earlier good fits.
  * Writes the full standard output set for the winning model only.

  python fastopic_best.py --input hustle_core.csv --num-topics 10 --seeds 5 --out-dir out_best
"""
import re, json, argparse, pickle, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

URL=re.compile(r"http\S+|www\.\S+"); HANDLE=re.compile(r"[@#]\w+")
NL=re.compile(r"\\n|\\r|[\r\n]+"); NONAL=re.compile(r"[^a-z0-9\s]"); WS=re.compile(r"\s+")
EXTRA_STOP=set("""
subscribe subscribed subscribing channel channels video videos watch watching like likes
comment comments follow following instagram twitter facebook snapchat link links bio click
free best top new ways way make making money get today guys guy hey welcome thanks
thank please enjoy hit bell notification share check vlog vlogs daily weekly episode part
com www http https youtube sub subs subscriber subscribers
trainings affiliat affilerator affliate durianriders kopywritingkourse zimaleta dsgenie
rockstarlivevideo clickfunn alfonso startabusinessforcheap affilia faceboo definite
visitor advertisements banners urls fridays backup marley slime psychic fraternity supreme
rich nomad growtopia die teachable cow wls broke star music artist art gta""".split())

def clean(t):
    t=str(t).lower(); t=NL.sub(" ",t); t=URL.sub(" ",t); t=HANDLE.sub(" ",t)
    return WS.sub(" ",NONAL.sub(" ",t)).strip()

def is_english(s):
    if not s: return False
    return sum(c.isascii() and c.isalpha() for c in s)/max(1,len(s))>0.6

def build_vocab(docs, vocab_size, min_df, max_df):
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    cv=CountVectorizer(stop_words=list(ENGLISH_STOP_WORDS|EXTRA_STOP), max_features=vocab_size,
                       min_df=min_df, max_df=max_df, token_pattern=r"(?u)\b[a-z]{3,}\b")
    cv.fit(docs); return set(cv.get_feature_names_out())

def fit_once(docs, K, top_words, device, seed):
    import torch; torch.manual_seed(seed); np.random.seed(seed)
    from fastopic import FASTopic
    kw=dict(num_topics=K, num_top_words=top_words, device=device, verbose=False,
            doc_embed_model="all-MiniLM-L6-v2")
    try: m=FASTopic(**kw)
    except TypeError: kw.pop("doc_embed_model",None); m=FASTopic(**kw)
    top, dt_ = m.fit_transform(docs)
    tw=[(t.split() if isinstance(t,str) else [str(w) for w in t])[:top_words] for t in top]
    return m, np.asarray(dt_), tw

def evaluate(tok_texts, topic_words, topn=10):
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    d=Dictionary(tok_texts); vocab=set(d.token2id)
    topics=[[w for w in t[:topn] if w in vocab] for t in topic_words]
    valid=[t for t in topics if len(t)>=2]
    per=[0.0]*len(topic_words)
    if len(valid)>=2:
        cm=CoherenceModel(topics=valid, texts=tok_texts, dictionary=d, coherence="c_npmi", topn=topn)
        vc=cm.get_coherence_per_topic(); k=0
        for i,t in enumerate(topics):
            if len(t)>=2: per[i]=float(vc[k]); k+=1
    flat=[w for t in topic_words for w in t[:topn]]
    div=len(set(flat))/max(1,len(flat))
    return per, div

def n_collapsed(topic_words, topn=10):
    sigs=[frozenset(t[:topn]) for t in topic_words]; seen=[]; c=0
    for s in sigs:
        if any(len(s&p)/max(1,len(s|p))>=0.8 for p in seen): c+=1
        else: seen.append(s)
    return c

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="hustle_core.csv"); ap.add_argument("--out-dir",default="out_best")
    ap.add_argument("--num-topics",type=int,default=10)
    ap.add_argument("--seeds",type=int,default=5)
    ap.add_argument("--min-tokens",type=int,default=3); ap.add_argument("--top-words",type=int,default=15)
    ap.add_argument("--vocab-size",type=int,default=30000); ap.add_argument("--min-df",type=int,default=5)
    ap.add_argument("--max-df",type=float,default=0.3)
    args=ap.parse_args()

    out=Path(args.out_dir); (out/"model").mkdir(parents=True,exist_ok=True)
    import torch; device="cuda" if torch.cuda.is_available() else "cpu"; print(f"[device] {device}")

    df=pd.read_csv(args.input,dtype=str,keep_default_na=False,engine="python"); print(f"[load] {len(df)} rows")
    title=df["title"].fillna("")
    df["doc"]=[clean(x) for x in title]; df["n_tok"]=df["doc"].str.split().str.len()
    eng=df["doc"].map(is_english); keep=(df["n_tok"]>=args.min_tokens)&eng
    df=df[keep].reset_index(drop=True)
    print(f"[clean] kept {len(df)} English docs ({(~eng).sum()} non-English dropped)")

    vocab=build_vocab(df["doc"].tolist(), args.vocab_size, args.min_df, args.max_df)
    print(f"[vocab] {len(vocab)} terms (min_df={args.min_df}, vocab_size={args.vocab_size})")
    docs=[" ".join(w for w in d.split() if w in vocab) for d in df["doc"].tolist()]
    nz=[i for i,d in enumerate(docs) if len(d.split())>=2]
    df=df.iloc[nz].reset_index(drop=True); docs=[docs[i] for i in nz]
    tok_texts=[d.split() for d in docs]
    print(f"[vocab] {len(docs)} docs with >=2 in-vocab words")

    # ---- multi-seed: fit, score, keep best (most coherent, fewest collapsed) ----
    best=None
    for s in range(args.seeds):
        seed=42+s*17
        m,doc_topic,tw=fit_once(docs,args.num_topics,args.top_words,device,seed)
        per,div=evaluate(tok_texts,tw); coll=n_collapsed(tw)
        good=[p for i,p in enumerate(per)]  # all topics
        mean_npmi=float(np.mean(per))
        # score: reward coherence + diversity, punish each collapsed topic hard
        score=mean_npmi + 0.1*div - 0.5*coll
        print(f"[seed {seed}] mean_npmi={mean_npmi:+.4f} diversity={div:.3f} collapsed={coll} score={score:+.4f}")
        if best is None or score>best["score"]:
            best=dict(seed=seed,model=m,doc_topic=doc_topic,tw=tw,per=per,div=div,coll=coll,
                      mean_npmi=mean_npmi,score=score)
    print(f"[best] seed={best['seed']} mean_npmi={best['mean_npmi']:+.4f} "
          f"diversity={best['div']:.3f} collapsed={best['coll']}")

    model=best["model"]; doc_topic=best["doc_topic"]; topic_words=best["tw"]
    per_topic_npmi=best["per"]; diversity=best["div"]; K=doc_topic.shape[1]
    df["topic"]=doc_topic.argmax(1); df["topic_prob"]=doc_topic.max(1)

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sid=SentimentIntensityAnalyzer()
    df["vader"]=[sid.polarity_scores(t)["compound"] for t in df["title"].fillna("")]
    df["views"]=pd.to_numeric(df["view_count"],errors="coerce").fillna(0)
    df["quarter"]=pd.to_datetime(df["upload_date"],dayfirst=True,errors="coerce").dt.to_period("Q").astype(str)

    tax=[]
    for k in range(K):
        sub=df[df["topic"]==k]; ex=sub.sort_values("topic_prob",ascending=False)["title"].head(6).tolist()
        tax.append({"topic_id":k,"size":int(len(sub)),
            "view_share":float(sub["views"].sum()/max(1,df["views"].sum())),
            "mean_vader":round(float(sub["vader"].mean()) if len(sub) else 0,3),
            "npmi":round(per_topic_npmi[k],3),"top_words":" ".join(topic_words[k]),
            "examples":" || ".join(ex)})
    tax_df=pd.DataFrame(tax).sort_values("size",ascending=False); tax_df.to_csv(out/"topic_taxonomy.csv",index=False)
    with open(out/"topic_inspection.txt","w",encoding="utf-8") as f:
        for _,r in tax_df.iterrows():
            f.write(f"TOPIC {r['topic_id']} (n={r['size']}, views={r['view_share']:.1%}, npmi={r['npmi']}, sent={r['mean_vader']})\n  {r['top_words']}\n")
            for e in r["examples"].split(" || "): f.write(f"   - {e}\n")
            f.write("\n")

    qorder=sorted(df["quarter"].dropna().unique()); tc=[]; tv=[]
    for q in qorder:
        m_=(df["quarter"]==q).values; idx=np.where(m_)[0]
        if not len(idx): continue
        w=df.loc[m_,"views"].values
        tc.append([q,int(len(idx))]+doc_topic[idx].mean(0).tolist())
        tv.append([q,float(w.sum())]+((doc_topic[idx]*w[:,None]).sum(0)/max(1.,w.sum())).tolist())
    cols=["quarter","n_docs"]+[f"topic_{k}" for k in range(K)]
    pd.DataFrame(tc,columns=cols).to_csv(out/"topic_trajectories.csv",index=False)
    pd.DataFrame(tv,columns=["quarter","total_views"]+[f"topic_{k}" for k in range(K)]).to_csv(out/"topic_trajectories_views.csv",index=False)
    df.groupby(["topic","quarter"])["vader"].mean().reset_index().to_csv(out/"topic_sentiment.csv",index=False)
    df[["display_id","channel_id","upload_date","quarter","topic","topic_prob","vader","views","niches"]].to_parquet(out/"doc_topics.parquet",index=False)

    np.save(out/"model"/"doc_topic_dist.npy",doc_topic); json.dump(topic_words,open(out/"model"/"topic_words.json","w"))
    try: model.save(str(out/"model"/"fastopic.zip"))
    except Exception:
        try: pickle.dump(model,open(out/"model"/"fastopic.pkl","wb"))
        except Exception as e: print(f"[save] {e}")
    json.dump({"n_docs":int(len(df)),"num_topics":K,"text":"title","best_seed":best["seed"],
        "npmi_mean":best["mean_npmi"],"npmi_per_topic":per_topic_npmi,"topic_diversity":diversity,
        "n_collapsed":best["coll"],"min_df":args.min_df,"max_df":args.max_df,"vocab_terms":len(vocab),
        "seeds_tried":args.seeds,"device":device,
        "finished":dt.datetime.now(dt.timezone.utc).isoformat()},open(out/"eval_metrics.json","w"),indent=2)
    print(f"[done] best model written -> {out}/")

if __name__=="__main__": main()
