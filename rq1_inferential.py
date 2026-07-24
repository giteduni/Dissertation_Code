#!/usr/bin/env python3
"""
rq1_inferential.py — inferential analysis for RQ1 (run LOCALLY).

Turns the descriptive K=6 taxonomy into defensible, citable claims:

  (A) CHANGE-POINT DETECTION  — PELT on each strategy's quarterly upload-share.
      * penalty chosen by a BIC-style rule, with a sensitivity sweep reported
      * each breakpoint annotated with date, pre/post mean shift, and Cohen's d
      * small-n early quarters guarded by a minimum-videos-per-quarter floor
  (B) ATTENTION CONCENTRATION — per-strategy Gini on per-video views,
      each with a 95% bootstrap CI (quantifies the production/attention decoupling)
  (C) DECOUPLING TEST         — Spearman rho between upload-share and view-share
      ranks across strategies (formalises "what is made != what is watched")

Outputs (./rq1_analysis/):
  changepoints.csv            one row per detected break (strategy, quarter, shift, d)
  changepoint_sensitivity.csv breakpoint count vs penalty (audit trail)
  gini.csv                    per-strategy Gini + bootstrap CI + medians/means
  decoupling.txt              Spearman result + interpretation
  fig_changepoints.pdf/.png   per-strategy share with detected breaks marked
  fig_gini_lorenz.pdf/.png    Lorenz curves + Gini bars
  summary.md                  copy-paste-ready findings for the methods/results chapter

    pip install pandas numpy matplotlib ruptures scipy pyarrow
    python rq1_inferential.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ----------------------------------------------------------------------------- CONFIG
RUN_DIR = Path("C:/Users/natha/OneDrive - University of Edinburgh/Dissertation/Data/YouNiverse/code/inference")                      # folder holding doc_topics.parquet
OUT     = Path("rq1_analysis"); OUT.mkdir(exist_ok=True)
MIN_Q_VIDEOS = 15                        # floor: ignore quarters with < this many videos (small-n noise)
BOOT = 2000                              # bootstrap resamples for Gini CI
SEED = 42

# topic_id -> (name, super-category). EDIT to match YOUR run's topic_taxonomy.csv.
TOPIC_NAMES = {
    5: ("Trading & Day-Trading",            "Markets"),
    1: ("E-commerce (Amazon/Shopify)",      "E-commerce"),
    2: ("Entrepreneurship & Real Estate",   "Business"),
    0: ("Online Earning / Freelance / WFH", "Services"),
    # 3,4 collapsed filler -> excluded
}
SUPERCAT_COLOURS = {"E-commerce":"#2E7D32","Markets":"#1565C0","Business":"#6A1B9A",
                    "Services":"#EF6C00","Content":"#AD1457","Other":"#546E7A"}
KEEP    = list(TOPIC_NAMES.keys())
name_of = lambda t: TOPIC_NAMES.get(t,(f"topic_{t}","Other"))[0]
col_of  = lambda t: SUPERCAT_COLOURS.get(TOPIC_NAMES.get(t,(None,"Other"))[1],"#546E7A")

# documented real-world events for external validation (edit/extend with citations)
EVENTS = {
    "2017Q4":"Bitcoin run-up to ~$20k (late 2017 crypto bubble)",
    "2016Q3":"Dropshipping/Shopify surge (2016–17 e-commerce boom)",
    "2020Q1":"COVID-19 onset (out of YouNiverse range; for Commons extension)",
}

mpl.rcParams.update({"font.family":"serif","font.size":11,"axes.titlesize":13,
    "axes.spines.top":False,"axes.spines.right":False,
    "figure.dpi":120,"savefig.dpi":300,"savefig.bbox":"tight"})
def save(fig,n): fig.savefig(OUT/f"{n}.pdf"); fig.savefig(OUT/f"{n}.png"); plt.close(fig); print(f"  wrote {n}")

# ----------------------------------------------------------------------------- LOAD
docs = pd.read_parquet(RUN_DIR/"doc_topics.parquet")
docs = docs[docs["topic"].isin(KEEP)].dropna(subset=["quarter"]).copy()
docs["views"] = pd.to_numeric(docs["views"], errors="coerce").fillna(0)
# quarterly upload-share per kept strategy, with small-n floor
qcount = docs.groupby("quarter").size()
valid_q = qcount[qcount>=MIN_Q_VIDEOS].index
d = docs[docs["quarter"].isin(valid_q)]
counts = d.groupby(["quarter","topic"]).size().unstack(fill_value=0).reindex(columns=KEEP,fill_value=0)
share  = counts.div(counts.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)
quarters = list(share.index)
print(f"[load] {len(docs)} classified videos; {len(quarters)} quarters >= {MIN_Q_VIDEOS} videos")

def q_to_x(q): return quarters.index(q)

# ============================================================ (A) CHANGE-POINT (PELT)
def pelt_breaks(series, pen):
    import ruptures as rpt
    algo = rpt.Pelt(model="rbf").fit(series.reshape(-1,1))
    return algo.predict(pen=pen)[:-1]          # drop the final index (end sentinel)

def cohens_d(a,b):
    na,nb=len(a),len(b)
    if na<2 or nb<2: return np.nan
    sp=np.sqrt(((na-1)*np.var(a,ddof=1)+(nb-1)*np.var(b,ddof=1))/(na+nb-2))
    return (np.mean(b)-np.mean(a))/sp if sp>0 else np.nan

def analyse_changepoints():
    import ruptures as rpt
    rows=[]; sens=[]
    # BIC-style penalty: pen = c * log(n) * sigma^2 ; report a sweep around it
    n=len(quarters)
    for t in KEEP:
        s=share[t].values.astype(float); sigma2=np.var(s)
        base_pen=max(1e-3, np.log(n)*sigma2*6)         # 6x BIC: report dominant breaks, not every step on a trend
        bks=pelt_breaks(s, base_pen)
        for b in bks:
            pre,post=s[:b],s[b:]
            rows.append({"strategy":name_of(t),"break_index":int(b),
                "break_quarter":quarters[b] if b<n else None,
                "pre_mean":round(float(np.mean(pre)),3),"post_mean":round(float(np.mean(post)),3),
                "shift":round(float(np.mean(post)-np.mean(pre)),3),
                "cohens_d":round(float(cohens_d(pre,post)),2),
                "direction":"rise" if np.mean(post)>np.mean(pre) else "decline"})
        # penalty sensitivity audit
        for mult in (1,2,3,5,8):
            sens.append({"strategy":name_of(t),"pen_mult_xBIC":mult,
                         "n_breaks":len(pelt_breaks(s, np.log(n)*sigma2*mult))})
    cp=pd.DataFrame(rows); cp.to_csv(OUT/"changepoints.csv",index=False)
    pd.DataFrame(sens).to_csv(OUT/"changepoint_sensitivity.csv",index=False)
    print(f"[A] {len(cp)} change-points detected (6x-BIC penalty)")
    return cp

def fig_changepoints(cp):
    ncol=2; nrow=int(np.ceil(len(KEEP)/ncol))
    fig,axes=plt.subplots(nrow,ncol,figsize=(12,3*nrow),sharex=True)
    xs=range(len(quarters))
    for i,t in enumerate(KEEP):
        a=axes.flat[i]; s=share[t].values
        a.fill_between(xs,s,color=col_of(t),alpha=.25)
        a.plot(xs,s,color=col_of(t),lw=1.5)
        for _,r in cp[cp["strategy"]==name_of(t)].iterrows():
            bi=r["break_index"]
            a.axvline(bi,color="black",ls="--",lw=1)
            a.annotate(f"{r['break_quarter']}\nΔ={r['shift']:+.2f} (d={r['cohens_d']})",
                       (bi,a.get_ylim()[1]*0.92),fontsize=7,ha="center",
                       bbox=dict(boxstyle="round,pad=0.2",fc="white",ec="grey",lw=.5))
            if r["break_quarter"] in EVENTS:
                a.annotate("●",(bi,0.02),color="red",fontsize=10,ha="center")
        a.set_title(name_of(t),fontsize=11); a.set_ylabel("Upload share")
    for j in range(len(KEEP),nrow*ncol): axes.flat[j].axis("off")
    step=max(1,len(quarters)//6)
    for a in axes.flat[-ncol:]:
        a.set_xticks(list(xs)[::step]); a.set_xticklabels(quarters[::step],rotation=45,ha="right",fontsize=8)
    fig.suptitle("Detected structural breaks in strategy prevalence (PELT, RBF cost, 6x-BIC penalty)\n"
                 "● = coincides with a documented external event",y=1.01)
    fig.tight_layout(); save(fig,"fig_changepoints")

# ============================================================ (B) GINI CONCENTRATION
def gini(x):
    x=np.sort(np.asarray(x,dtype=float)); n=len(x)
    if n==0 or x.sum()==0: return np.nan
    return float((2*np.arange(1,n+1)-n-1).dot(x)/(n*x.sum()))

def gini_ci(x,boot=BOOT,seed=SEED):
    rng=np.random.default_rng(seed); x=np.asarray(x,dtype=float)
    bs=[gini(rng.choice(x,size=len(x),replace=True)) for _ in range(boot)]
    return np.nanpercentile(bs,2.5),np.nanpercentile(bs,97.5)

def analyse_gini():
    rows=[]
    for t in KEEP:
        v=docs[docs["topic"]==t]["views"].values
        g=gini(v); lo,hi=gini_ci(v)
        rows.append({"strategy":name_of(t),"n_videos":int(len(v)),
            "gini_views":round(g,3),"gini_ci_low":round(lo,3),"gini_ci_high":round(hi,3),
            "median_views":int(np.median(v)),"mean_views":int(np.mean(v)),
            "top1pct_view_share":round(float(np.sort(v)[-max(1,len(v)//100):].sum()/max(1,v.sum())),3)})
    gdf=pd.DataFrame(rows).sort_values("gini_views",ascending=False)
    gdf.to_csv(OUT/"gini.csv",index=False)
    print(f"[B] Gini computed (overall mean = {gdf['gini_views'].mean():.3f})")
    return gdf

def fig_gini(gdf):
    fig,ax=plt.subplots(1,2,figsize=(12,4.6))
    # Lorenz curves
    for t in KEEP:
        v=np.sort(docs[docs["topic"]==t]["views"].values.astype(float))
        if v.sum()==0: continue
        cum=np.insert(np.cumsum(v)/v.sum(),0,0); x=np.linspace(0,1,len(cum))
        ax[0].plot(x,cum,color=col_of(t),lw=1.8,label=name_of(t))
    ax[0].plot([0,1],[0,1],"k--",lw=.8,label="perfect equality")
    ax[0].set_xlabel("Cumulative share of videos"); ax[0].set_ylabel("Cumulative share of views")
    ax[0].set_title("Lorenz curves: view concentration by strategy"); ax[0].legend(fontsize=8,frameon=False)
    # Gini bars with CI
    g=gdf.sort_values("gini_views")
    cols=[col_of([k for k,vv in TOPIC_NAMES.items() if vv[0]==n][0]) for n in g["strategy"]]
    err=[g["gini_views"]-g["gini_ci_low"], g["gini_ci_high"]-g["gini_views"]]
    ax[1].barh(g["strategy"],g["gini_views"],xerr=err,color=cols,capsize=4)
    ax[1].set_xlim(0,1); ax[1].set_xlabel("Gini coefficient of views (95% bootstrap CI)")
    ax[1].set_title("Attention inequality within each strategy")
    fig.tight_layout(); save(fig,"fig_gini_lorenz")

# ============================================================ (C) DECOUPLING TEST
def analyse_decoupling(gdf):
    from scipy.stats import spearmanr
    upload=counts.sum(axis=0).reindex(KEEP); upload.index=[name_of(t) for t in KEEP]
    viewsum=docs.groupby("topic")["views"].sum().reindex(KEEP); viewsum.index=[name_of(t) for t in KEEP]
    rho,p=spearmanr(upload.values,viewsum.values)
    txt=(f"Production-attention decoupling (Spearman across {len(KEEP)} strategies)\n"
         f"  upload volume vs total views:  rho = {rho:+.3f}, p = {p:.3f}\n\n"
         f"  Interpretation: {'weak/!no' if abs(rho)<0.5 else 'moderate-strong'} rank correlation -> "
         f"what is produced most is {'NOT' if abs(rho)<0.5 else ''} what is watched most.\n")
    (OUT/"decoupling.txt").write_text(txt, encoding="utf-8"); print("[C] decoupling test written")
    return rho,p

# ============================================================ SUMMARY
def write_summary(cp,gdf,rho,p):
    lines=["# RQ1 inferential findings (auto-generated)\n",
           f"_Quarters analysed: {len(quarters)} (>= {MIN_Q_VIDEOS} videos/quarter). "
           f"Strategies: {', '.join(name_of(t) for t in KEEP)}._\n",
           "## A. Structural breaks (PELT, RBF cost, 6x-BIC penalty)\n"]
    for _,r in cp.iterrows():
        ev=EVENTS.get(r["break_quarter"]); evtxt=f"  *(coincides with: {ev})*" if ev else ""
        lines.append(f"- **{r['strategy']}** — {r['direction']} at **{r['break_quarter']}**, "
                     f"mean share {r['pre_mean']}→{r['post_mean']} (Δ {r['shift']:+.2f}, Cohen's d = {r['cohens_d']}).{evtxt}")
    lines.append("\n## B. Attention concentration (Gini on views, 95% bootstrap CI)\n")
    for _,r in gdf.iterrows():
        lines.append(f"- **{r['strategy']}**: Gini = {r['gini_views']} "
                     f"[{r['gini_ci_low']}, {r['gini_ci_high']}]; "
                     f"top-1% of videos hold {r['top1pct_view_share']:.0%} of views; "
                     f"median {r['median_views']:,} vs mean {r['mean_views']:,} views.")
    lines.append(f"\n## C. Production–attention decoupling\n")
    lines.append(f"- Spearman rho (upload volume vs total views) = **{rho:+.3f}** (p = {p:.3f}) — "
                 f"{'decoupled: production rank does not predict attention rank' if abs(rho)<0.5 else 'correlated'}.")
    lines.append("\n_Penalty sensitivity in `changepoint_sensitivity.csv`; figures in this folder._")
    (OUT/"summary.md").write_text("\n".join(lines), encoding="utf-8"); print("  wrote summary.md")

# ----------------------------------------------------------------------------- RUN
if __name__=="__main__":
    cp=analyse_changepoints(); fig_changepoints(cp)
    gdf=analyse_gini();        fig_gini(gdf)
    rho,p=analyse_decoupling(gdf)
    write_summary(cp,gdf,rho,p)
    print(f"\nDone. See {OUT}/summary.md and the figures.")
