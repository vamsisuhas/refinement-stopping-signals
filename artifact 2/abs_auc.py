import json, re, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

MAX_ITERS = 4
blob = json.load(open('trajectories.json'))
trajs = blob['trajectories'] if isinstance(blob, dict) else blob

FAMILIES = {"A:confidence":[0,1], "B:stability":[2,3,4,7,8], "C:critique":[6,9,10], "D:history":[5,11,12]}

def step_features(t, i):
    st=t["steps"]; cur=st[i]
    conf=cur["confidence"]/100.0
    conf_prev=st[i-1]["confidence"]/100.0 if i>0 else conf
    n_changes=sum(s["changed"] for s in st[max(0,i-2):i+1])
    cs=[s["canon"] for s in st[:i+1]]
    osc=int(len(cs)>=3 and cs[-1]==cs[-3] and cs[-1]!=cs[-2])
    streak=0
    for s in reversed(st[1:i+1]):
        if s["changed"]==0: streak+=1
        else: break
    crit=cur.get("critique_text") or ""; sol=cur.get("full_text") or ""
    sol_prev=(st[i-1].get("full_text") or "") if i>0 else sol
    return [conf, conf-conf_prev, cur["changed"], n_changes, osc, i/MAX_ITERS,
            cur["critique_error"], len(set(cs))/(i+1.0), streak/MAX_ITERS,
            min(len(crit),2000)/2000.0, int(bool(re.search(r"\d",crit))),
            (len(sol)-len(sol_prev))/1000.0, int(t["kind"]=="math")]

X,Yimp,Yharm,Q=[],[],[],[]
for qid,t in enumerate(trajs):
    st=t["steps"]
    for i in range(len(st)-1):
        X.append(step_features(t,i))
        Yimp.append(int(st[i]["correct"]==0 and st[i+1]["correct"]==1))
        Yharm.append(int(st[i]["correct"]==1 and st[i+1]["correct"]==0))
        Q.append(qid)
X=np.array(X); Yimp=np.array(Yimp); Yharm=np.array(Yharm); Q=np.array(Q)
Xc=X[:, FAMILIES["A:confidence"]]
print(f"decision points {len(Q)} | improvement {Yimp.sum()} | harm {Yharm.sum()}")

def oof_rank(Xa,y,n_splits=4,seed=0):
    sc=np.full(len(y),np.nan); pos=set(Q[y==1].tolist())
    if len(pos)<2: return sc
    skf=StratifiedGroupKFold(n_splits=int(min(n_splits,len(pos))),shuffle=True,random_state=seed)
    for tr,te in skf.split(Xa,y,groups=Q):
        if len(set(y[tr].tolist()))<2: continue
        p=LogisticRegression(max_iter=1000).fit(Xa[tr],y[tr]).predict_proba(Xa[te])[:,1]
        sc[te]=rankdata(p)/(len(p)+1.0)
    return sc

def mauc(y,s):
    m=~np.isnan(s)
    return roc_auc_score(y[m],s[m]) if len(set(y[m].tolist()))>1 else float('nan')

# question-level clustered bootstrap of the ABSOLUTE AUC (the paper's own CI method)
def boot_abs(y,s,n_boot=2000,seed=0):
    rng=np.random.default_rng(seed); qids=np.unique(Q); vals=[]
    for _ in range(n_boot):
        samp=qids[rng.integers(0,len(qids),len(qids))]
        rows=np.concatenate([np.where(Q==q)[0] for q in samp])
        yy,ss=y[rows],s[rows]; m=~np.isnan(ss)
        if len(set(yy[m].tolist()))<2: continue
        vals.append(roc_auc_score(yy[m],ss[m]))
    lo,hi=np.percentile(vals,[2.5,97.5]); return float(lo),float(hi)

print("\n=== ABSOLUTE AUC vs chance (0.5), question-level clustered bootstrap, 2000 resamples ===")
for name,y in [("improvement (ALL 13 feats)",Yimp),("harm (ALL 13 feats)",Yharm)]:
    s=oof_rank(X,y); pt=mauc(y,s); lo,hi=boot_abs(y,s)
    verdict = "ABOVE chance" if lo>0.5 else ("indistinguishable from chance" if hi>0.5 else "BELOW chance")
    print(f"  {name:32s} AUC={pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  -> {verdict}")
for fam,idx in FAMILIES.items():
    s=oof_rank(X[:,idx],Yharm); pt=mauc(Yharm,s); lo,hi=boot_abs(Yharm,s)
    print(f"  harm, {fam:20s} AUC={pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
