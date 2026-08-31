import json, re, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
blob=json.load(open('/Users/vamsisuhas/Downloads/refinement_experiments/trajectories.json'))
trajs=blob['trajectories']; MAX_ITERS=4

def norm_text(s):
    s=re.sub(r"[-_]"," ",s.lower()); s=re.sub(r"[^a-z0-9 ]","",s)
    return re.sub(r"\s+"," ",s).strip()

# SHARED canonicalizer for both domains (kills the extractor confound)
def canon_shared(a): return norm_text(a)

# bootstrap CI on the critic-verdict harm AUC
verd,y_cur,harm,Q=[],[],[],[]
for qid,t in enumerate(trajs):
    st=t["steps"]
    for i in range(len(st)-1):
        verd.append(st[i+1]["critique_error"]); y_cur.append(st[i]["correct"])
        harm.append(int(st[i]["correct"]==1 and st[i+1]["correct"]==0)); Q.append(qid)
verd,y_cur,harm,Q=map(np.array,(verd,y_cur,harm,Q))
m=y_cur==1
def boot_auc(y,s,q,n=2000,seed=0):
    rng=np.random.default_rng(seed); qs=np.unique(q); v=[]
    for _ in range(n):
        samp=qs[rng.integers(0,len(qs),len(qs))]
        rows=np.concatenate([np.where(q==x)[0] for x in samp])
        if len(set(y[rows].tolist()))<2: continue
        v.append(roc_auc_score(y[rows],s[rows]))
    return float(np.percentile(v,2.5)), float(np.percentile(v,97.5))
lo,hi=boot_auc(harm[m],verd[m],Q[m])
print(f"critic verdict -> harmful flip (among currently-correct): AUC={roc_auc_score(harm[m],verd[m]):.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

# ---- de-confounded transfer: shared canonicalizer for all trajectory features ----
def feats(t,i,canon):
    st=t["steps"]; cur=st[i]
    cs=[canon(s["answer"]) for s in st[:i+1]]
    changed=[0]+[int(cs[j]!=cs[j-1]) for j in range(1,len(cs))]
    conf=cur["confidence"]/100.0
    conf_prev=st[i-1]["confidence"]/100.0 if i>0 else conf
    n_ch=sum(changed[max(0,i-2):i+1])
    osc=int(len(cs)>=3 and cs[-1]==cs[-3] and cs[-1]!=cs[-2])
    streak=0
    for c in reversed(changed[1:i+1]):
        if c==0: streak+=1
        else: break
    crit=cur.get("critique_text") or ""; sol=cur.get("full_text") or ""
    solp=(st[i-1].get("full_text") or "") if i>0 else sol
    return [conf, conf-conf_prev, changed[i], n_ch, osc, i/MAX_ITERS, cur["critique_error"],
            len(set(cs))/(i+1.0), streak/MAX_ITERS, min(len(crit),2000)/2000.0,
            int(bool(re.search(r"\d",crit))), (len(sol)-len(solp))/1000.0]

for label,canon in [("ORIGINAL (per-domain extractors)", None), ("SHARED canonicalizer", canon_shared)]:
    X,Yi,Yh,K,QQ=[],[],[],[],[]
    for qid,t in enumerate(trajs):
        st=t["steps"]
        for i in range(len(st)-1):
            if canon is None:
                cs=[s["canon"] for s in st[:i+1]]
                f=feats(t,i,lambda a,_st=st: None) if False else None
            X.append(feats(t,i,canon) if canon else feats(t,i,lambda a: a))
            Yi.append(int(st[i]["correct"]==0 and st[i+1]["correct"]==1))
            Yh.append(int(st[i]["correct"]==1 and st[i+1]["correct"]==0))
            K.append(t["kind"]); QQ.append(qid)
    X=np.array(X); Yi=np.array(Yi); Yh=np.array(Yh); K=np.array(K); QQ=np.array(QQ)
    print(f"\n=== TRANSFER, {label} ===")
    for a,b in [("math","qa"),("qa","math")]:
        tr,te=K==a,K==b
        for nm,y in [("improve",Yi),("harm",Yh)]:
            if len(set(y[tr].tolist()))<2 or len(set(y[te].tolist()))<2: continue
            mdl=LogisticRegression(max_iter=1000).fit(X[tr],y[tr])
            s=mdl.predict_proba(X[te])[:,1]
            pt=roc_auc_score(y[te],s); l,h=boot_auc(y[te],s,QQ[te])
            print(f"  train {a:>4} -> test {b:<4} {nm:8s} AUC={pt:.3f}  95% CI [{l:.3f}, {h:.3f}]")
