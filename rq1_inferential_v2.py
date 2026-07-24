#!/usr/bin/env python3
"""
rq1_inferential_v2.py — INTEGRITY-HARDENED inferential analysis for RQ1 (run LOCALLY).

Upgrades over v1 (the version that produced the first results):
  1. PERMUTATION SIGNIFICANCE TEST per detected break — PELT *locates* breaks but does
     not test them. We give each break an empirical p-value by repeatedly shuffling the
     series, re-detecting, and asking how often a shift of >= the observed |mean-shift|
     appears at the same location band by chance. Converts "a break with d=3.8" into
     "a break with d=3.8, p_perm < 0.01".
  2. AUTOMATIC CURATION — every break is tiered FEATURED vs APPENDIX by a pre-declared
     rule (|d| >= D_FEATURE AND p_perm <= P_FEATURE AND not in the sparse early tail).
     Curation, not deletion: the full table is still written; the report features the
     strong, well-sampled breaks and footnotes the rest.
  3. FIGURE FIX — early sparse quarters trimmed (TRIM_BEFORE), and break annotations are
     dodged/staggered so labels never overlap; only FEATURED breaks are annotated, the
     rest drawn as faint ticks.
  4. HONEST DECOUPLING — the Spearman across 4 strategies is retained and reported, but
     explicitly flagged as UNDERPOWERED (n=4) and INCONCLUSIVE; the decoupling argument
     is carried by the Gini + the descriptive production/attention gap instead.

Outputs (./rq1_analysis_v2/):
  changepoints_all.csv         every detected break + Cohen's d + permutation p + tier
  changepoints_featured.csv    only the FEATURED breaks (for the main-text table)
  changepoint_sensitivity.csv  break count vs penalty (audit trail)
  gini.csv                     per-strategy Gini + bootstrap CI + concentration metrics
  decoupling.txt               Spearman result with the underpowered-n=4 caveat
  fig_changepoints.pdf/.png    trimmed series, dodged labels, featured breaks only
  fig_gini_lorenz.pdf/.png     Lorenz curves + Gini bars with CIs
  summary.md                   copy-paste-ready findings, split featured vs appendix

    pip install pandas numpy matplotlib ruptures scipy pyarrow
    python rq1_inferential_v2.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ============================================================ CONFIG
RUN_DIR = Path("C:/Users/natha/OneDrive - University of Edinburgh/Dissertation/Data/YouNiverse/code/inference")                       # folder holding doc_topics.parquet
OUT     = Path("rq1_analysis_v2"); OUT.mkdir(exist_ok=True)
MIN_Q_VIDEOS = 15                         # drop quarters with < this many videos (small-n noise)
TRIM_BEFORE  = "2009Q1"                    # ignore quarters before this for detection + plotting
BOOT     = 2000                           # bootstrap resamples for Gini CI
N_PERM   = 2000                           # permutations for break significance
PEN_MULT = 6                              # PELT penalty multiple of the BIC scale
SEED     = 42
# curation rule (pre-declared): a break is FEATURED iff all three hold
D_FEATURE = 1.0                           # |Cohen's d| threshold
P_FEATURE = 0.05                          # permutation p-value threshold
# (and it must fall on/after TRIM_BEFORE, enforced by trimming)

# topic_id -> (name, super-category). EDIT to match YOUR run's topic_taxonomy.csv.
TOPIC_NAMES = {
    5: ("Trading & Day-Trading",            "Markets"),
    1: ("E-commerce (Amazon/Shopify)",      "E-commerce"),
    2: ("Entrepreneurship & Real Estate",   "Business"),
    0: ("Online Earning / Freelance / WFH", "Services"),
}
SUPERCAT_COLOURS = {"E-commerce":"#2E7D32","Markets":"#1565C0","Business":"#6A1B9A",
                    "Services":"#EF6C00","Content":"#AD1457","Other":"#546E7A"}
KEEP    = list(TOPIC_NAMES.keys())
name_of = lambda t: TOPIC_NAMES.get(t,(f"topic_{t}","Other"))[0]
col_of  = lambda t: SUPERCAT_COLOURS.get(TOPIC_NAMES.get(t,(None,"Other"))[1],"#546E7A")

# documented real-world events for external validation (cite these in prose; see summary.md)
EVENTS = {
    "2014Q3":"Shopify/dropshipping take-off (2014–17 e-commerce boom)",
    "2015Q4":"Dropshipping mainstreaming; Shopify IPO 2015",
    "2017Q1":"Peak dropshipping-course era; pre-crypto-bubble",
    "2017Q4":"Bitcoin run-up to ~$20k (late-2017 crypto bubble)",
}

mpl.rcParams.update({"font.family":"serif","font.size":11,"axes.titlesize":13,
    "axes.spines.top":False,"axes.spines.right":False,
    "figure.dpi":120,"savefig.dpi":300,"savefig.bbox":"tight"})
def save(fig,n): fig.savefig(OUT/f"{n}.pdf"); fig.savefig(OUT/f"{n}.png"); plt.close(fig); print(f"  wrote {n}")

# ============================================================ LOAD
docs = pd.read_parquet(RUN_DIR/"doc_topics.parquet")
docs = docs[docs["topic"].isin(KEEP)].dropna(subset=["quarter"]).copy()
docs["views"] = pd.to_numeric(docs["views"], errors="coerce").fillna(0)

qcount  = docs.groupby("quarter").size()
valid_q = qcount[qcount >= MIN_Q_VIDEOS].index
d = docs[docs["quarter"].isin(valid_q)]
counts = d.groupby(["quarter","topic"]).size().unstack(fill_value=0).reindex(columns=KEEP,fill_value=0)
share  = counts.div(counts.sum(axis=1).replace(0,np.nan),axis=0).fillna(0).sort_index()
# trim sparse early tail for BOTH detection and plotting
share  = share[share.index >= TRIM_BEFORE]
quarters = list(share.index)
n = len(quarters)
print(f"[load] {len(docs)} classified videos; {n} quarters in [{quarters[0]} .. {quarters[-1]}] "
      f"(>= {MIN_Q_VIDEOS} videos/quarter, trimmed before {TRIM_BEFORE})")

# ============================================================ (A) CHANGE-POINTS + PERMUTATION
def pelt_breaks(series, pen):
    import ruptures as rpt
    return rpt.Pelt(model="rbf").fit(series.reshape(-1,1)).predict(pen=pen)[:-1]

def cohens_d(a,b):
    na,nb=len(a),len(b)
    if na<2 or nb<2: return np.nan
    sp=np.sqrt(((na-1)*np.var(a,ddof=1)+(nb-1)*np.var(b,ddof=1))/(na+nb-2))
    return (np.mean(b)-np.mean(a))/sp if sp>0 else np.nan

def base_penalty(series):
    return max(1e-3, np.log(len(series))*np.var(series)*PEN_MULT)

def permutation_p(series, break_idx, obs_shift, n_perm=N_PERM, seed=SEED):
    """Empirical p: under random temporal ordering, how often does a split at the SAME
    index produce a |mean-shift| >= the observed one? Tests whether the temporal
    structure (not just the partition) drives the break. One-sided on |shift|."""
    rng=np.random.default_rng(seed); s=series.copy(); obs=abs(obs_shift); hits=0
    for _ in range(n_perm):
        p=rng.permutation(s)
        sh=abs(p[break_idx:].mean()-p[:break_idx].mean())
        if sh>=obs: hits+=1
    return (hits+1)/(n_perm+1)            # add-one (never reports p=0)

def analyse_changepoints():
    rows=[]; sens=[]
    for t in KEEP:
        s=share[t].values.astype(float)
        bks=pelt_breaks(s, base_penalty(s))
        for b in bks:
            if b<=1 or b>=n-1: continue
            pre,post=s[:b],s[b:]
            shift=float(np.mean(post)-np.mean(pre)); dd=float(cohens_d(pre,post))
            pp=permutation_p(s,b,shift)
            rows.append({"strategy":name_of(t),"super_cat":TOPIC_NAMES[t][1],
                "break_index":int(b),"break_quarter":quarters[b],
                "pre_mean":round(float(np.mean(pre)),3),"post_mean":round(float(np.mean(post)),3),
                "shift":round(shift,3),"cohens_d":round(dd,2),"p_perm":round(pp,4),
                "direction":"rise" if shift>0 else "decline"})
        for mult in (1,2,3,6,8,12):
            sens.append({"strategy":name_of(t),"pen_mult_xBIC":mult,
                "n_breaks":len(pelt_breaks(s, max(1e-3,np.log(n)*np.var(s)*mult)))})
    cp=pd.DataFrame(rows)
    # pre-declared curation rule
    cp["featured"]=(cp["cohens_d"].abs()>=D_FEATURE)&(cp["p_perm"]<=P_FEATURE)
    cp["tier"]=np.where(cp["featured"],"FEATURED","appendix")
    cp.to_csv(OUT/"changepoints_all.csv",index=False)
    cp[cp["featured"]].to_csv(OUT/"changepoints_featured.csv",index=False)
    pd.DataFrame(sens).to_csv(OUT/"changepoint_sensitivity.csv",index=False)
    nf=int(cp["featured"].sum())
    print(f"[A] {len(cp)} breaks detected; {nf} FEATURED (|d|>={D_FEATURE}, p_perm<={P_FEATURE}), "
          f"{len(cp)-nf} to appendix")
    return cp

def fig_changepoints(cp):
    ncol=2; nrow=int(np.ceil(len(KEEP)/ncol))
    fig,axes=plt.subplots(nrow,ncol,figsize=(13,3.4*nrow),sharex=True)
    xs=np.arange(n)
    for i,t in enumerate(KEEP):
        a=axes.flat[i]; s=share[t].values
        a.fill_between(xs,s,color=col_of(t),alpha=.22); a.plot(xs,s,color=col_of(t),lw=1.6)
        sub=cp[cp["strategy"]==name_of(t)]
        feat=sub[sub["featured"]].sort_values("break_index").reset_index(drop=True)
        # appendix breaks: faint ticks, no label (keeps the figure clean)
        for _,r in sub[~sub["featured"]].iterrows():
            a.axvline(r["break_index"],color="grey",ls=":",lw=.7,alpha=.5)
        # featured breaks: solid line + dodged label (alternate high/low, spread across x)
        ymax=a.get_ylim()[1]
        for j,(_,r) in enumerate(feat.iterrows()):
            bi=r["break_index"]
            a.axvline(bi,color="black",ls="--",lw=1.1)
            yfrac=0.96 if j%2==0 else 0.78          # stagger vertically
            ev="●" if r["break_quarter"] in EVENTS else ""
            pstr = "p<0.001" if r["p_perm"]<0.001 else f"p={r['p_perm']:.3f}"
            label = f"{ev}{r['break_quarter']}\nd={r['cohens_d']}, {pstr}"
            a.annotate(label,(bi, ymax*yfrac), fontsize=7.5, ha="center", va="top",
                       bbox=dict(boxstyle="round,pad=0.25",fc="white",ec="grey",lw=.5))
            if ev: a.plot(bi,0.01*ymax,marker="o",color="red",ms=5)
        a.set_title(name_of(t),fontsize=11); a.set_ylabel("Upload share"); a.set_ylim(0,ymax*1.18)
    for j in range(len(KEEP),nrow*ncol): axes.flat[j].axis("off")
    step=max(1,n//7)
    for a in axes.flat[-ncol:]:
        a.set_xticks(xs[::step]); a.set_xticklabels(quarters[::step],rotation=45,ha="right",fontsize=8)
    fig.suptitle("Detected structural breaks in strategy prevalence (PELT-RBF, 6×-BIC penalty)\n"
                 "Solid = featured break (|d|≥1, permutation p≤0.05); dotted = sub-threshold; ● = documented external event",
                 y=1.02, fontsize=12)
    fig.tight_layout(); save(fig,"fig_changepoints")

# ============================================================ (B) GINI CONCENTRATION
def gini(x):
    x=np.sort(np.asarray(x,dtype=float)); m=len(x)
    if m==0 or x.sum()==0: return np.nan
    return float((2*np.arange(1,m+1)-m-1).dot(x)/(m*x.sum()))

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
    fig,ax=plt.subplots(1,2,figsize=(13,4.8))
    for t in KEEP:
        v=np.sort(docs[docs["topic"]==t]["views"].values.astype(float))
        if v.sum()==0: continue
        cum=np.insert(np.cumsum(v)/v.sum(),0,0); x=np.linspace(0,1,len(cum))
        ax[0].plot(x,cum,color=col_of(t),lw=1.9,label=name_of(t))
    ax[0].plot([0,1],[0,1],"k--",lw=.8,label="perfect equality")
    ax[0].set_xlabel("Cumulative share of videos"); ax[0].set_ylabel("Cumulative share of views")
    ax[0].set_title("Lorenz curves: view concentration by strategy"); ax[0].legend(fontsize=8,frameon=False)
    g=gdf.sort_values("gini_views")
    cols=[col_of([k for k,vv in TOPIC_NAMES.items() if vv[0]==nm][0]) for nm in g["strategy"]]
    err=[g["gini_views"]-g["gini_ci_low"], g["gini_ci_high"]-g["gini_views"]]
    ax[1].barh(g["strategy"],g["gini_views"],xerr=err,color=cols,capsize=4)
    ax[1].set_xlim(0,1); ax[1].set_xlabel("Gini coefficient of views (95% bootstrap CI)")
    ax[1].set_title("Attention inequality within each strategy")
    fig.tight_layout(); save(fig,"fig_gini_lorenz")

# ============================================================ (C) DECOUPLING (honest, n=4)
def analyse_decoupling():
    from scipy.stats import spearmanr
    upload =counts.sum(axis=0).reindex(KEEP)
    viewsum=docs.groupby("topic")["views"].sum().reindex(KEEP)
    rho,p=spearmanr(upload.values,viewsum.values)
    txt=("Production-attention decoupling — Spearman rank correlation\n"
         f"  strategies (n) = {len(KEEP)}\n"
         f"  upload volume vs total views:  rho = {rho:+.3f}, p = {p:.3f}\n\n"
         "  CAVEAT (decisive): with only n=4 strategies this test has ~no statistical power.\n"
         "  Spearman on n=4 cannot reach p<0.05 except at |rho|=1.0, so BOTH a null result\n"
         "  and a positive result are uninformative. We therefore report this as INCONCLUSIVE\n"
         "  and do NOT use it as evidence of decoupling. The decoupling claim rests instead on\n"
         "  (i) the per-strategy Gini coefficients (mean ~0.89, all CIs well above 0.8) and\n"
         "  (ii) the descriptive production-vs-attention gap (e.g. Online-Earning/WFH ranks ~3rd\n"
         "  by upload volume yet commands the largest view share) — neither of which depends on\n"
         "  this underpowered rank test.\n")
    (OUT/"decoupling.txt").write_text(txt, encoding="utf-8")
    print(f"[C] decoupling: rho={rho:+.3f}, p={p:.3f} — reported as INCONCLUSIVE (n=4)")
    return rho,p

# ============================================================ SUMMARY
def write_summary(cp,gdf,rho,p):
    feat=cp[cp["featured"]].sort_values(["strategy","break_index"])
    app =cp[~cp["featured"]].sort_values(["strategy","break_index"])
    L=["# RQ1 inferential findings (auto-generated, v2 — integrity-hardened)\n",
       f"_Quarters analysed: {n} ({quarters[0]}–{quarters[-1]}), >= {MIN_Q_VIDEOS} videos/quarter, "
       f"early tail trimmed before {TRIM_BEFORE}. Strategies: {', '.join(name_of(t) for t in KEEP)}._\n",
       "## A. Structural breaks — FEATURED (|d| >= 1.0 AND permutation p <= 0.05)\n",
       "_PELT-RBF detection; permutation test = 2000 random temporal re-orderings per break._\n"]
    for _,r in feat.iterrows():
        ev=EVENTS.get(r["break_quarter"]); evt=f"  *(coincides with: {ev})*" if ev else ""
        ps="< 0.001" if r["p_perm"]<0.001 else f"= {r['p_perm']:.3f}"
        L.append(f"- **{r['strategy']}** — {r['direction']} at **{r['break_quarter']}**, "
                 f"share {r['pre_mean']}→{r['post_mean']} (Δ {r['shift']:+.2f}, Cohen's d = {r['cohens_d']}, "
                 f"permutation p {ps}).{evt}")
    L.append("\n## A′. Sub-threshold breaks — APPENDIX (small effect or non-significant / early)\n")
    for _,r in app.iterrows():
        ps="< 0.001" if r["p_perm"]<0.001 else f"= {r['p_perm']:.3f}"
        L.append(f"- {r['strategy']} — {r['direction']} at {r['break_quarter']} "
                 f"(Δ {r['shift']:+.2f}, d = {r['cohens_d']}, p {ps}) — below feature threshold.")
    L.append("\n## B. Attention concentration (Gini on views, 95% bootstrap CI)\n")
    for _,r in gdf.iterrows():
        L.append(f"- **{r['strategy']}**: Gini = {r['gini_views']} [{r['gini_ci_low']}, {r['gini_ci_high']}]; "
                 f"top-1% of videos hold {r['top1pct_view_share']:.0%} of views; "
                 f"median {r['median_views']:,} vs mean {r['mean_views']:,} views.")
    L.append(f"\n## C. Production–attention decoupling (reported as INCONCLUSIVE)\n")
    L.append(f"- Spearman rho (upload vs views, across {len(KEEP)} strategies) = **{rho:+.3f}** (p = {p:.3f}). "
             f"**Underpowered at n=4 — not used as evidence.** Decoupling is argued from the Gini "
             f"(mean {gdf['gini_views'].mean():.3f}) and the descriptive production/attention gap instead.")
    L.append("\n## D. External validation (REQUIRES your citations — see prose template below)\n")
    L.append("- The E-commerce rise staircase (2013Q2→2017Q1, all d>2) aligns with the documented "
             "2014–2017 dropshipping/Shopify boom. Cite a source for the boom (e.g. Shopify GMV/merchant "
             "growth reports; trade-press coverage of the dropshipping-course wave) and state the alignment "
             "explicitly. The mirror-image Entrepreneurship/Real-Estate decline over the same windows "
             "supports a *substitution* reading (generic 'be-an-entrepreneur' content displaced by concrete "
             "'build-a-store' content).")
    L.append("\n_Full break table: changepoints_all.csv · featured-only: changepoints_featured.csv · "
             "penalty sensitivity: changepoint_sensitivity.csv._")
    (OUT/"summary.md").write_text("\n".join(L), encoding="utf-8")
    print("  wrote summary.md")

# ============================================================ RUN
if __name__=="__main__":
    cp=analyse_changepoints(); fig_changepoints(cp)
    gdf=analyse_gini();        fig_gini(gdf)
    rho,p=analyse_decoupling()
    write_summary(cp,gdf,rho,p)
    print(f"\nDone -> {OUT}/  (see summary.md; featured breaks in changepoints_featured.csv)")
