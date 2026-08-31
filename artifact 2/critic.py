import json, re, numpy as np
from sklearn.metrics import roc_auc_score
blob=json.load(open('trajectories.json'))
trajs=blob['trajectories']; MAX_ITERS=4

# ---- 1. What does the critic's BINARY verdict actually track? ----
# cur["critique_error"] at step i+1 = the critic's verdict on answer i (1 = flagged an error)
verd, y_cur, imp, harm = [], [], [], []
for t in trajs:
    st=t["steps"]
    for i in range(len(st)-1):
        verd.append(st[i+1]["critique_error"])       # verdict on answer i
        y_cur.append(st[i]["correct"])
        imp.append(int(st[i]["correct"]==0 and st[i+1]["correct"]==1))
        harm.append(int(st[i]["correct"]==1 and st[i+1]["correct"]==0))
verd=np.array(verd); y_cur=np.array(y_cur); imp=np.array(imp); harm=np.array(harm)

print("=== CRITIC'S BINARY VERDICT: what does it track? ===")
print(f"  flags an error on {verd.mean():.1%} of decision points ({verd.sum()}/{len(verd)})")
print(f"  flag rate | current answer WRONG : {verd[y_cur==0].mean():.3f}")
print(f"  flag rate | current answer RIGHT : {verd[y_cur==1].mean():.3f}")
print(f"  AUC(verdict -> current answer is wrong)  [Q1] = {roc_auc_score(1-y_cur, verd):.3f}")
print(f"  AUC(verdict -> next step fixes it)  among wrong = {roc_auc_score(imp[y_cur==0], verd[y_cur==0]):.3f}")
print(f"  AUC(verdict -> next step breaks it) among right = {roc_auc_score(harm[y_cur==1], verd[y_cur==1]):.3f}")

# ---- 2. Permutation null for the judge policy ----
def judge_stop(st, verds):
    for i in range(len(st)-1):
        if verds[i]==0: return i
    return len(st)-1

def replay(perm=None, rng=None):
    if perm is not None: pool=perm.copy(); rng.shuffle(pool)
    k=0; corr=0; iters=0
    for t in trajs:
        st=t["steps"]; n=len(st)-1
        v=[st[i+1]["critique_error"] for i in range(n)] if perm is None else list(pool[k:k+n])
        k+=n
        s=judge_stop(st,v); corr+=st[s]["correct"]; iters+=s
    return corr/len(trajs), iters/len(trajs)

obs_acc, obs_it = replay()
rng=np.random.default_rng(0)
null=[replay(verd, rng) for _ in range(1000)]
na=np.array([x[0] for x in null]); ni=np.array([x[1] for x in null])
lo,hi=np.percentile(na,[2.5,97.5])
print("\n=== PERMUTATION NULL: is the critic's verdict coupled to the specific answer? ===")
print(f"  observed judge policy : acc={obs_acc:.3f}  iters={obs_it:.2f}")
print(f"  verdict-shuffled null : acc={na.mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}]  iters={ni.mean():.2f}")
p=float((na>=obs_acc).mean())
print(f"  observed sits at percentile {100*(na<obs_acc).mean():.1f} of the null; one-sided p={p:.3f}")
print(f"  -> {'NOT distinguishable from a verdict carrying no answer-specific information' if lo<=obs_acc<=hi else 'distinguishable from the decoupled null'}")
