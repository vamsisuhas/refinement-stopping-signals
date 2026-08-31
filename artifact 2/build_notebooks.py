#!/usr/bin/env python3
"""Generate 4 Colab notebooks (one per hypothesis) for the
'Confidence Is Not a Stopping Rule' self-refinement study, using NVIDIA Nemotron 3.

v2 — hardened after an adversarial multi-agent review:
  * real Nemotron 3 model ids (verified live on integrate.api.nvidia.com, July 2026)
  * thinking mode disabled (Nemotron 3 defaults to reasoning-on, which eats max_tokens)
  * fail-fast API/key validation; chat() raises instead of returning ""
  * cache stamping + resume; never persists an empty cache
  * MATH-500 (level>=4, numeric golds) as the default dataset; embedded set is fallback only
  * H2 AUCs conditioned on current correctness (paper Q1 vs Q2)
  * H3/H4 use group-stratified CV with within-fold rank normalization / base-rate
    calibration (pooled LOQO probabilities are systematically biased with rare positives)
  * external-judge baseline + token cost + fix-rate + regret in H4; cross-dataset transfer cell
  * question-level bootstrap CIs behind every SUPPORTED/NOT SUPPORTED/INCONCLUSIVE verdict
  * minimum-event guards (UNTESTABLE instead of confident-looking noise)
"""
import json, os

OUT = "."
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# SHARED HARNESS  (identical top of every notebook)
# ----------------------------------------------------------------------------
SHARED = r'''# ============================================================
#  SHARED HARNESS  —  self-refinement loop on NVIDIA Nemotron 3
#  (this same block is at the top of all 4 hypothesis notebooks)
# ============================================================
!pip -q install "openai>=1.30" datasets scikit-learn scipy matplotlib numpy

import os, re, json, time, getpass
import numpy as np

# ---------- 1. NVIDIA API key (free from https://build.nvidia.com) ----------
API_KEY = os.environ.get("NVIDIA_API_KEY", "")
if not API_KEY:
    API_KEY = getpass.getpass("Paste your NVIDIA API key (nvapi-...): ").strip()

from openai import OpenAI, APIStatusError, BadRequestError
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=API_KEY)

# ---------- 2. Model + experiment config ----------
# Nemotron 3 family, hosted free on build.nvidia.com (ids verified live, July 2026).
# The small nano model is the default on purpose: it is fast, cheap on credits, and
# wrong often enough that refinement transitions actually occur (a huge model that is
# right at iteration 0 on every question leaves nothing to study).
MODEL = "nvidia/nemotron-3-nano-30b-a3b"
#   larger Nemotron 3 alternatives (same endpoint):
#   "nvidia/nemotron-3-super-120b-a12b"
#   "nvidia/nemotron-3-ultra-550b-a55b"
#   "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

MAX_ITERS  = 4      # refinement steps after the initial answer (paper suggests up to five)
N_EXAMPLES = 446    # 223 math (all usable MATH-500 level>=3 numeric items) + 223 multi-hop QA;
                    # 9 API calls each -> ~4000 calls (~2-2.5 h free tier; resumable, saves as it goes)
RATE_SLEEP = 1.2    # seconds between API calls (free tier allows roughly 40 requests/min per model)

# Refinement-prompt style. The paper (sec. 18) explicitly asks to compare both, because the
# revision instruction itself can suppress or amplify the transitions being studied:
#   "neutral"      — the revision is free to change or keep the answer
#   "conservative" — the revision is told to keep an already-correct answer
PROMPT_STYLE = "neutral"

# Record mean token log-probabilities per response (paper sec. 11A "token-level
# probability" signal). Read-only: does not change model behavior. If the endpoint
# rejects the parameter, capture switches itself off and the run continues.
CAPTURE_LOGPROBS = True

# Cache filename includes the prompt style so a style-ablation run can NEVER overwrite
# an existing run — both live side by side.
TRAJ_PATH = f"trajectories_{PROMPT_STYLE}.json"

# Persist the cache to Google Drive so a Colab runtime crash/recycle loses NOTHING, and
# all four notebooks automatically share the same file (no download/upload dance).
# Set to False to keep the file on the runtime's local (ephemeral!) disk instead.
USE_DRIVE = True
if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        _ddir = "/content/drive/MyDrive/refinement_experiments"
        os.makedirs(_ddir, exist_ok=True)
        TRAJ_PATH = os.path.join(_ddir, f"trajectories_{PROMPT_STYLE}.json")
        _legacy = os.path.join(_ddir, "trajectories.json")
        if PROMPT_STYLE == "neutral" and not os.path.exists(TRAJ_PATH) and os.path.exists(_legacy):
            os.rename(_legacy, TRAJ_PATH)   # adopt a pre-existing cache as the neutral run
            print("adopted existing trajectories.json as", TRAJ_PATH)
        print("trajectories will be saved to Google Drive:", TRAJ_PATH)
    except ImportError:
        pass    # not running in Colab — keep the local path

# Benchmark data. Default pulls MATH-500 (level>=4, plain-numeric answers) from HuggingFace.
# The EMBEDDED list further below is an offline smoke-test fallback ONLY — it is far too easy:
# a capable model answers ~everything at iteration 0, so there are no wrong->right or
# right->wrong transitions left for H1-H4 to measure.
USE_HF_MATH500 = True
USE_HF_HOTPOT  = True    # multi-hop QA with context paragraphs (longer prompts, slower)

MIN_EVENTS = 10     # below this many transition events, AUC-based verdicts are meaningless

# ---------- 3. One robust chat call ----------
_extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
_last_lp = {"v": None}   # mean token logprob of the most recent chat() response

def chat(messages, temperature=0.3, max_tokens=1024):
    """Nemotron 3 models 'think' by default: the reasoning trace consumes max_tokens and the
    final content can come back truncated or empty. We disable thinking explicitly; if a
    non-Nemotron-3 model rejects the flag we drop it and retry. Raises on unrecoverable
    errors instead of returning "" (a silent "" would poison every statistic downstream)."""
    global _extra, CAPTURE_LOGPROBS
    last = None
    time.sleep(RATE_SLEEP)
    for attempt in range(5):
        try:
            params = dict(model=MODEL, messages=messages, temperature=temperature,
                          top_p=0.95, max_tokens=max_tokens, **_extra)
            if CAPTURE_LOGPROBS:
                params["logprobs"] = True
            r = client.chat.completions.create(**params)
            txt = r.choices[0].message.content or ""
            # some servings inline the trace in content instead of message.reasoning_content
            txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
            tok = r.usage.total_tokens if r.usage else 0
            _last_lp["v"] = None
            try:
                lptoks = r.choices[0].logprobs.content
                if lptoks:
                    _last_lp["v"] = float(np.mean([tk.logprob for tk in lptoks]))
            except (AttributeError, TypeError):
                pass
            return txt, tok
        except BadRequestError:
            if CAPTURE_LOGPROBS:       # endpoint rejects logprobs -> continue without them
                CAPTURE_LOGPROBS = False
                print("   (endpoint rejected logprobs; continuing without logprob capture)")
                continue
            if _extra:                 # model rejects chat_template_kwargs -> drop it, retry
                _extra = {}
                continue
            raise                      # a plain bad request will not fix itself
        except APIStatusError as e:
            if e.status_code in (401, 403, 404):
                raise                  # bad key / bad model id — retrying cannot help
            last = e
            wait = 10 * (attempt + 1)  # 429 / 5xx: back off harder
        except Exception as e:
            last = e
            wait = 3 * (attempt + 1)
        print(f"   API retry {attempt+1} ({last}); sleeping {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"NVIDIA API failed after 5 attempts: {last}")

def validate_api():
    """Fail fast with an actionable message on a bad key or dead model id."""
    try:
        chat([{"role": "user", "content": "Reply with the single word OK."}], max_tokens=8)
        print(f"API OK — model {MODEL} responds.")
    except APIStatusError as e:
        if e.status_code in (401, 403):
            raise SystemExit("NVIDIA API key rejected. Get a free key at https://build.nvidia.com "
                             "(open any model page -> Get API Key), then re-run this cell.")
        if e.status_code == 404:
            raise SystemExit(f"Model id {MODEL!r} was not found on the endpoint. Switch MODEL to one "
                             "of the alternatives listed in the config above, then re-run this cell.")
        raise

# ---------- 4. Parse ANSWER / CONFIDENCE out of a response ----------
def parse_answer(text):
    hits = re.findall(r"ANSWER:\s*(.+)", text, re.I)
    if hits:  # take the LAST occurrence — drafts inside reasoning text come earlier
        return re.split(r"(?i)\s*CONFIDENCE\s*:", hits[-1])[0].strip()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""          # empty content must never raise

def parse_confidence(text):
    """Returns (confidence 0-100, parsed_ok). Handles '85', '85%', '92.5' and 0-1 scales."""
    hits = re.findall(r"CONFIDENCE:\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    if not hits:
        return 50.0, 0
    raw = hits[-1]
    v = float(raw)
    if v <= 1.0 and "." in raw:                # model answered on a 0-1 scale
        v *= 100.0
    return max(0.0, min(100.0, v)), 1

def norm_num(s):
    s = s.replace(",", "").replace("$", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    return float(nums[-1]) if nums else None

def norm_text(s):
    s = re.sub(r"[-_]", " ", s.lower())        # 'Stratford-upon-Avon' == 'stratford upon avon'
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()

def canon(ans, kind):
    """One canonical form used consistently for 'changed', stability, and oscillation,
    so '88' vs '88.0' never counts as an answer change."""
    if kind == "math":
        v = norm_num(ans)
        return str(v) if v is not None else norm_text(ans)
    return norm_text(ans)

def is_correct(pred, gold, kind):
    if kind == "math":
        p, g = norm_num(pred), norm_num(gold)
        return (p is not None and g is not None and abs(p - g) < 1e-2)
    p, g = norm_text(pred), norm_text(gold)
    return len(g) > 1 and len(p) > 0 and (g in p or p in g)

# ---------- 5. The three prompts of a refinement loop ----------
SOLVE_SYS = ("You are a careful problem solver. Think briefly, then finish with EXACTLY two lines:\n"
             "ANSWER: <your final answer, as short as possible>\n"
             "CONFIDENCE: <integer 0-100 = your probability that the answer is correct>")

REVISE_TAIL = {
    "neutral":      "Provide your best solution to the problem, taking the critique into account only if it is valid.",
    "conservative": "Give an improved solution. If the previous answer was already correct, keep it.",
}

def solve(q):
    txt, tok = chat([{"role": "system", "content": SOLVE_SYS},
                     {"role": "user", "content": q}], temperature=0.3)
    conf, cok = parse_confidence(txt)
    return txt, parse_answer(txt), conf, cok, tok, _last_lp["v"]

def critique(q, solution):
    sys = ("You are a strict reviewer. Read the problem and the proposed solution. "
           "Point out any specific error, or say 'No error found.' Be concise (<=4 sentences).")
    txt, tok = chat([{"role": "system", "content": sys},
                     {"role": "user", "content": f"PROBLEM:\n{q}\n\nSOLUTION:\n{solution}"}],
                    temperature=0.5)
    return txt, tok

def revise(q, solution, crit):
    txt, tok = chat([{"role": "system", "content": SOLVE_SYS},
                     {"role": "user", "content":
                      f"PROBLEM:\n{q}\n\nPREVIOUS SOLUTION:\n{solution}\n\n"
                      f"REVIEWER CRITIQUE:\n{crit}\n\n" + REVISE_TAIL[PROMPT_STYLE]}],
                    temperature=0.4)
    conf, cok = parse_confidence(txt)
    return txt, parse_answer(txt), conf, cok, tok, _last_lp["v"]

def critique_flags_error(crit):
    """A critique 'identifies an error' if it is non-empty and not a clean bill of health."""
    return bool(crit.strip()) and not re.search(
        r"no error|is correct|looks correct|already correct", crit, re.I)

# ---------- 6. Build one trajectory for one question ----------
def build_traj(ex):
    q, gold, kind = ex["question"], ex["answer"], ex["kind"]
    full, ans, conf, cok, tok, lp = solve(q)
    steps = [{"iter": 0, "answer": ans, "canon": canon(ans, kind), "confidence": conf,
              "conf_parsed": cok, "correct": int(is_correct(ans, gold, kind)),
              "changed": 0, "critique_error": 0, "tokens": tok,
              "full_text": full, "critique_text": "", "logprob_mean": lp}]
    for i in range(1, MAX_ITERS + 1):
        crit, tc = critique(q, full)
        full, ans, conf, cok, tr, lp = revise(q, full, crit)
        c = canon(ans, kind)
        steps.append({"iter": i, "answer": ans, "canon": c, "confidence": conf,
                      "conf_parsed": cok, "correct": int(is_correct(ans, gold, kind)),
                      "changed": int(c != steps[-1]["canon"]),
                      "critique_error": int(critique_flags_error(crit)),
                      "tokens": tc + tr, "full_text": full, "critique_text": crit,
                      "logprob_mean": lp})
    return {"question": q, "gold": gold, "kind": kind, "steps": steps}

# ---------- 7. Datasets ----------
# EMBEDDED = offline smoke-test fallback ONLY (see the note in the config section).
EMBEDDED = [
 {"kind":"math","answer":"88","question":"A store offers a 20% discount on a $100 item and then adds 10% sales tax. What is the final price in dollars?"},
 {"kind":"math","answer":"40","question":"If a train travels 60 miles in 1.5 hours, what is its average speed in miles per hour?"},
 {"kind":"math","answer":"36","question":"What is 15% of 240?"},
 {"kind":"math","answer":"25","question":"The sum of three consecutive integers is 72. What is the largest of the three?"},
 {"kind":"math","answer":"50","question":"A shirt costs $40 after a 20% discount. What was the original price in dollars?"},
 {"kind":"math","answer":"2","question":"What is the remainder when 2^10 is divided by 7?"},
 {"kind":"math","answer":"9","question":"How many positive divisors does 36 have?"},
 {"kind":"math","answer":"110","question":"Compute the sum of the first 10 positive even numbers."},
 {"kind":"math","answer":"6","question":"If a fair die is rolled twice, in how many ordered ways can the two rolls sum to 7?"},
 {"kind":"math","answer":"42","question":"What is 7 factorial divided by 5 factorial?"},
 {"kind":"math","answer":"5","question":"What is the logarithm base 2 of 32?"},
 {"kind":"math","answer":"15","question":"A right triangle has legs of length 9 and 12. What is the length of the hypotenuse?"},
 {"kind":"math","answer":"11","question":"The average of 4 numbers is 10. Three of them are 8, 12, and 9. What is the fourth number?"},
 {"kind":"math","answer":"18","question":"A tank fills at 5 liters per minute. How many minutes to fill 90 liters?"},
 {"kind":"qa","answer":"Stratford-upon-Avon","question":"In which English town was the author of 'Romeo and Juliet' born? Answer with the town name only."},
 {"kind":"qa","answer":"Tokyo","question":"What is the capital of the country whose official currency is the yen? Answer with the city name only."},
 {"kind":"qa","answer":"Pacific","question":"What is the largest ocean on Earth? One word."},
 {"kind":"qa","answer":"Einstein","question":"Which physicist formulated the theory of general relativity? Surname only."},
 {"kind":"qa","answer":"France","question":"In which country is the Eiffel Tower located? Country name only."},
 {"kind":"qa","answer":"gold","question":"The chemical element with atomic number 79 is commonly used for wedding rings. Name the element."},
]

def load_examples(n):
    exs = []
    try:
        if USE_HF_MATH500:
            from datasets import load_dataset
            import random
            ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
            # keep moderately-hard-and-up items whose gold is a plain number — norm_num
            # cannot grade LaTeX golds like \frac{3}{4} (it would extract 4.0 and silently
            # mis-score them). level>=3 numeric = 223 items; the easier level-3 tier also
            # supplies initially-correct answers, i.e. chances to observe harmful flips.
            rows = [r for r in ds if int(r["level"]) >= 3
                    and re.fullmatch(r"-?\d+(?:\.\d+)?", str(r["answer"]).strip())]
            random.Random(0).shuffle(rows)          # fixed seed => resumable order
            take = n if not USE_HF_HOTPOT else (n + 1) // 2
            exs += [{"kind": "math", "answer": str(r["answer"]).strip(),
                     "question": r["problem"] + "\nGive the final answer as a single plain number."}
                    for r in rows[:take]]
        if USE_HF_HOTPOT:
            from datasets import load_dataset
            take = n - len(exs)
            ds = load_dataset("hotpotqa/hotpot_qa", "distractor",
                              split=f"validation[:{max(take * 2, 50)}]")
            for r in ds:
                if len(exs) >= n:
                    break
                # include the context paragraphs — without them this is closed-book
                # trivia, not the multi-hop reading task the paper asks for
                ctx = " ".join(" ".join(s) for s in r["context"]["sentences"])[:4000]
                exs.append({"kind": "qa", "answer": str(r["answer"]),
                            "question": f"Context:\n{ctx}\n\nQuestion: {r['question']}\nAnswer briefly."})
    except Exception as e:
        print(f"NOTE: benchmark download failed ({e}); falling back to the embedded smoke-test set.")
    if not exs:
        print("NOTE: using the EMBEDDED smoke-test set — too easy for real hypothesis testing; "
              "expect UNTESTABLE/INCONCLUSIVE verdicts. Enable USE_HF_MATH500 for a real run.")
        exs = list(EMBEDDED)
    return exs[:n]

# ---------- 8. Sanity + statistics helpers (used by every notebook) ----------
def sanity_check(trajs):
    if not trajs:
        return
    it0 = float(np.mean([t["steps"][0]["correct"] for t in trajs]))
    trans = sum(a["correct"] != b["correct"]
                for t in trajs for a, b in zip(t["steps"], t["steps"][1:]))
    bad_conf = sum(1 - s["conf_parsed"] for t in trajs for s in t["steps"])
    print(f"[sanity] iteration-0 accuracy = {it0:.2f} | correctness transitions = {trans} "
          f"| unparsed confidences = {bad_conf}")
    if it0 > 0.9 or trans < MIN_EVENTS:
        print("[sanity] *** WARNING: too few wrong answers / transitions. H1 has no headroom and "
              "H2-H4 will be UNTESTABLE. Keep USE_HF_MATH500=True, raise N_EXAMPLES, and/or "
              "keep the small nano model rather than a larger one. ***")
    if bad_conf > len(trajs):
        print("[sanity] *** WARNING: many CONFIDENCE lines failed to parse (defaulted to 50). "
              "Inspect a step's full_text in trajectories.json before trusting H1/H2. ***")

def boot_ci(stat_fn, trajs, n_boot=1000, seed=0):
    """95% CI by question-level (clustered) bootstrap: resample whole trajectories, because
    the 4 decision points inside one trajectory are strongly dependent."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = [trajs[i] for i in rng.integers(0, len(trajs), len(trajs))]
        v = stat_fn(sample)
        if v == v:                      # drop NaN resamples (e.g. single-class)
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)

def verdict(name, lo, hi, null=0.0):
    if lo != lo:
        return f"{name}: UNTESTABLE on this run (no valid bootstrap resamples)"
    if lo > null:
        return f"{name}: SUPPORTED (95% question-level-bootstrap CI [{lo:+.3f}, {hi:+.3f}] excludes {null})"
    if hi < null:
        return f"{name}: NOT SUPPORTED (95% question-level-bootstrap CI [{lo:+.3f}, {hi:+.3f}] is below {null})"
    return (f"{name}: INCONCLUSIVE at this sample size "
            f"(95% question-level-bootstrap CI [{lo:+.3f}, {hi:+.3f}] spans {null}) — raise N_EXAMPLES")

# ---------- 9. Generate (or load cached) trajectories ----------
def _config():
    return {"model": MODEL, "max_iters": MAX_ITERS, "prompt_style": PROMPT_STYLE}

def get_trajectories(regenerate=False):
    done = []
    if os.path.exists(TRAJ_PATH) and not regenerate:
        try:
            blob = json.load(open(TRAJ_PATH))
        except Exception:
            blob = None                                    # corrupt / interrupted save
        if isinstance(blob, dict) and blob.get("config") == _config():
            done = blob.get("trajectories", [])
        elif blob is not None:
            print(f"Cached {TRAJ_PATH} was built with a different MODEL/MAX_ITERS/PROMPT_STYLE — regenerating.")
        if done and not all(len(t["steps"]) == MAX_ITERS + 1 for t in done):
            print(f"Cached {TRAJ_PATH} is inconsistent with MAX_ITERS={MAX_ITERS} — regenerating.")
            done = []
        if len(done) >= N_EXAMPLES:
            print(f"Loading cached {TRAJ_PATH} ({len(done)} trajectories; delete the file or "
                  f"call get_trajectories(regenerate=True) to rerun).")
            sanity_check(done)
            return done
    data = load_examples(N_EXAMPLES)
    # Resume by QUESTION TEXT, not by position: safe even if N_EXAMPLES, the dataset mix,
    # or the difficulty filter changed between sessions — already-run questions are kept,
    # only genuinely new ones are generated (no duplicates, no silent skips).
    have = {t["question"] for t in done}
    todo = [ex for ex in data if ex["question"] not in have]
    if not todo:
        print(f"Loading cached {TRAJ_PATH} ({len(done)} trajectories — nothing new to generate).")
        sanity_check(done)
        return done
    if done:
        print(f"Cache holds {len(done)} trajectories — generating {len(todo)} new ones.")
    validate_api()
    t0 = time.time()
    for j, ex in enumerate(todo):
        print(f"[{j+1}/{len(todo)}] ({ex['kind']}) {ex['question'][:55].strip()}...")
        try:
            done.append(build_traj(ex))
        except RuntimeError:
            raise                        # API/key/model is broken — abort loudly, keep partial cache
        except Exception as e:
            print("   skipped:", e)
        if done:                         # never persist an empty cache
            with open(TRAJ_PATH, "w") as f:
                json.dump({"config": _config(), "trajectories": done}, f, indent=1)
        rate = (time.time() - t0) / (j + 1)
        print(f"   elapsed {(time.time()-t0)/60:.1f} min, ~{(len(todo)-j-1)*rate/60:.1f} min left")
    if not done:
        raise RuntimeError("0 trajectories were generated — check your NVIDIA API key and MODEL, "
                           "then re-run this cell.")
    print(f"Done. Saved {len(done)} trajectories to {TRAJ_PATH}")
    sanity_check(done)
    return done

print("Harness ready. Model =", MODEL, "| prompt style =", PROMPT_STYLE)
'''

# ----------------------------------------------------------------------------
# Per-hypothesis markdown + analysis cells
# ----------------------------------------------------------------------------

H1_MD = r'''# H1 — Confidence Inflation
**Claim:** as the model critiques and revises, *reported confidence rises faster than
actual accuracy*. It feels more certain without becoming more correct.

**How we test it:** run the refinement loop, plot mean accuracy vs mean confidence per
iteration, count the four transition types between consecutive iterations
(wrong→right, right→wrong, right→right, wrong→wrong — the paper's transition-level
protocol), and put a **question-level bootstrap CI** around the inflation statistic
`(confidence rise) − (accuracy rise)` before printing a verdict.

> Run the SHARED HARNESS cell first. This notebook also **creates `trajectories.json`**,
> which H2/H3/H4 reuse — download it after this runs and upload it to the other notebooks
> so they don't pay to regenerate.'''

H1_CODE = r'''import matplotlib.pyplot as plt
trajs = get_trajectories()          # generates trajectories.json on first run
kinds = sorted(set(t["kind"] for t in trajs))

def curves(sub):
    acc = np.zeros(MAX_ITERS+1); conf = np.zeros(MAX_ITERS+1); cnt = np.zeros(MAX_ITERS+1)
    for t in sub:
        for s in t["steps"]:
            i = s["iter"]
            acc[i] += s["correct"]; conf[i] += s["confidence"]/100.0; cnt[i] += 1
    cnt = np.maximum(cnt, 1)
    return acc/cnt, conf/cnt

acc, conf = curves(trajs)
gap = conf - acc
print(f"{'iter':>4} {'accuracy':>9} {'confidence':>11} {'gap':>7}")
for i in range(MAX_ITERS+1):
    print(f"{i:>4} {acc[i]:>9.2f} {conf[i]:>11.2f} {gap[i]:>7.2f}")

# transition counts between consecutive iterations (paper analysis 2), with a
# question-level bootstrap CI on the NET value (fixes - breaks) of each round
print(f"\n{'step':>6} {'wrong->right':>13} {'right->wrong':>13} {'right->right':>13} "
      f"{'wrong->wrong':>13}   net value [95% CI]")
for i in range(MAX_ITERS):
    c = {"wr": 0, "rw": 0, "rr": 0, "ww": 0}
    for t in trajs:
        a, b = t["steps"][i]["correct"], t["steps"][i+1]["correct"]
        c[("r" if a else "w") + ("r" if b else "w")] += 1
    def net_i(sample, i=i):
        return float(np.mean([t["steps"][i+1]["correct"] - t["steps"][i]["correct"]
                              for t in sample]))
    nlo, nhi = boot_ci(net_i, trajs)
    print(f"{i}->{i+1:<3} {c['wr']:>13} {c['rw']:>13} {c['rr']:>13} {c['ww']:>13}   "
          f"{net_i(trajs):+.3f} [{nlo:+.3f}, {nhi:+.3f}]")

# confidence-saturation diagnosis: if confidence starts at ceiling, "inflation" is
# untestable by construction and saturation itself is the finding
allconf = np.array([s["confidence"] for t in trajs for s in t["steps"]])
sat = float((allconf >= 95).mean())
print(f"\n[saturation] {sat:.0%} of all reported confidences are >= 95 "
      f"(iteration-0 mean = {100*conf[0]:.0f}/100)")

xs = list(range(MAX_ITERS+1))
plt.figure(figsize=(8, 5))
plt.plot(xs, acc,  "o-", lw=2, ms=7, label="accuracy")
plt.plot(xs, conf, "s-", lw=2, ms=7, label="mean confidence")
plt.fill_between(xs, acc, conf, alpha=0.12, label="overconfidence gap")
plt.xticks(xs); plt.ylim(0, 1.05)
plt.xlabel("refinement iteration", fontsize=11); plt.ylabel("rate", fontsize=11)
plt.title("H1: does confidence inflate faster than accuracy?", fontsize=12)
plt.legend(loc="lower right", fontsize=10, frameon=False); plt.grid(alpha=.3)
plt.tight_layout(); plt.show()

def h1_stat(sample):
    a0 = np.mean([t["steps"][0]["correct"] for t in sample])
    a1 = np.mean([t["steps"][-1]["correct"] for t in sample])
    c0 = np.mean([t["steps"][0]["confidence"]/100 for t in sample])
    c1 = np.mean([t["steps"][-1]["confidence"]/100 for t in sample])
    return (c1 - c0) - (a1 - a0)

print(f"\n(confidence rise) - (accuracy rise), iteration 0 -> {MAX_ITERS}: {h1_stat(trajs):+.3f}")
for k in kinds:
    sub = [t for t in trajs if t["kind"] == k]
    klo, khi = boot_ci(h1_stat, sub)
    print(f"   {k:>4} only: {h1_stat(sub):+.3f} [95% CI {klo:+.3f}, {khi:+.3f}]  "
          f"(n={len(sub)} questions)")
lo, hi = boot_ci(h1_stat, trajs)
sat_regime = sat > 0.8 and conf[0] > 0.9
v = verdict("H1 (confidence inflation)", lo, hi)
if sat_regime and lo > 0:
    v += ("\n   [CAUTION: confidence is at ceiling, so this positive statistic is driven by the"
          "\n   accuracy term (accuracy flat or falling), NOT by confidence rising — do not quote"
          "\n   this line as evidence that confidence inflated]")
print(v)
if sat_regime:
    print("\n*** INTERPRETATION: confidence starts at ceiling (saturated), so the H1 dynamic")
    print("'confidence RISES faster than accuracy' is untestable by construction — there is")
    print("almost no room left to rise. The reportable finding is the saturation itself: the")
    print(f"model asserts ~{100*conf[0]:.0f}% confidence while its accuracy is {acc[0]:.0%}-{max(acc):.0%},")
    print("a constant overconfidence gap that makes confidence useless as a stopping signal")
    print("in this setup. ***")
print("\nNOTE: if iteration-0 accuracy is already ~1.0 (see the [sanity] line above), there is")
print("no headroom on the accuracy side either — use harder data before interpreting this verdict.")'''

H2_MD = r'''# H2 — Current Confidence Is Not Enough
**Claim:** the model's confidence in its *current* answer does **not** reliably predict
whether *one more* refinement step will improve the answer.

**How we test it:** the paper's core distinction (sec. 4) is Question 1 — *is the current
answer correct?* — vs Question 2 — *will another step help?*. So the **primary** metric is
conditioned: among currently-**wrong** answers only, does low confidence predict that the
next step fixes them? (And among currently-**right** answers, does high confidence predict
safety from harmful flips?) Without this conditioning, confidence can look predictive
merely because it tracks *current* correctness — Q1 leaking into a Q2 test, exactly the
Question A/B confusion from sec. 5 of the paper. A question-level bootstrap CI and a
minimum-event guard gate the verdict — and when the CI is *narrow* and sits inside
[0.45, 0.55], the notebook upgrades the verdict to an **equivalence-style bound**
(affirmative evidence that any confidence signal is at most trivial, not merely a
failure to detect one). A final saturation check reports what fraction of confidences
are ≥ 95, since a saturated signal cannot discriminate anything by construction.

> Upload the `trajectories.json` produced by the H1 notebook before running (otherwise
> this notebook regenerates it, costing a full API run).'''

H2_CODE = r'''from sklearn.metrics import roc_auc_score
trajs = get_trajectories()

conf_now, cur_right, improve, harm = [], [], [], []
for t in trajs:
    st = t["steps"]
    for i in range(len(st)-1):
        cur, nxt = st[i], st[i+1]
        conf_now.append(cur["confidence"])
        cur_right.append(cur["correct"])
        improve.append(int(cur["correct"]==0 and nxt["correct"]==1))  # wrong -> right
        harm.append(int(cur["correct"]==1 and nxt["correct"]==0))     # right -> wrong
conf_now = np.array(conf_now); cur_right = np.array(cur_right, dtype=bool)
improve = np.array(improve); harm = np.array(harm)
n_imp, n_harm = int(improve.sum()), int(harm.sum())
print(f"decision points: {len(conf_now)} | wrong->right: {n_imp} | right->wrong: {n_harm}")

def safe_auc(y, score):
    y = np.asarray(y)
    return roc_auc_score(y, score) if len(set(y.tolist())) > 1 else float("nan")

w, r = ~cur_right, cur_right
print("\nPRIMARY (paper's Q2, conditioned so Q1 'is it currently correct?' cannot leak in):")
print(f"  among currently-WRONG (n={int(w.sum())}):  AUC( -confidence -> next step FIXES it )  = "
      f"{safe_auc(improve[w], -conf_now[w]):.3f}   (0.5 = useless)")
print(f"  among currently-RIGHT (n={int(r.sum())}):  AUC(  confidence -> next step BREAKS it ) = "
      f"{safe_auc(harm[r], conf_now[r]):.3f}")
print(f"[secondary, unconditional] improve AUC {safe_auc(improve, -conf_now):.3f} | "
      f"harm AUC {safe_auc(harm, conf_now):.3f}  — these can look >0.5 merely because")
print("confidence tracks CURRENT correctness, which is why they are not the headline metric.")

print()
for lo_b, hi_b in [(0, 50), (50, 80), (80, 101)]:
    m = (conf_now >= lo_b) & (conf_now < hi_b)
    if m.sum():
        print(f"confidence [{lo_b:3d},{hi_b:3d}): improve-rate={improve[m].mean():.2f} "
              f"harm-rate={harm[m].mean():.2f}  (n={int(m.sum())})")

if min(n_imp, n_harm) < MIN_EVENTS:
    print(f"\nWARNING: only {n_imp} wrong->right and {n_harm} right->wrong events "
          f"(floor {MIN_EVENTS}, target >=30). AUCs above are statistically meaningless at")
    print("these counts — treat this run as a smoke test. Raise N_EXAMPLES / keep USE_HF_MATH500=True.")
    print("VERDICT: UNTESTABLE on this run.")
else:
    def h2_stat(sample):
        y, s = [], []
        for t in sample:
            st = t["steps"]
            for i in range(len(st)-1):
                if st[i]["correct"] == 0:                     # conditioned on currently-wrong
                    y.append(st[i+1]["correct"]); s.append(-st[i]["confidence"])
        return roc_auc_score(y, s) if len(set(y)) > 1 else float("nan")
    auc_pt = h2_stat(trajs)
    lo, hi = boot_ci(h2_stat, trajs)
    print(f"\nconditioned improvement AUC = {auc_pt:.3f}, 95% CI [{lo:.3f}, {hi:.3f}] "
          f"(question-level bootstrap)")
    for k in sorted(set(t["kind"] for t in trajs)):
        sub = [t for t in trajs if t["kind"] == k]
        print(f"   [descriptive] {k} only: conditioned improvement AUC = {h2_stat(sub):.3f} "
          f"(n={len(sub)} questions)")
    bound = max(hi, 1 - lo)   # AUC-equivalent of the largest |AUC - 0.5| in the CI
    if lo > 0.5 or hi < 0.5:
        msg = "H2 NOT SUPPORTED: confidence carries SOME next-step signal (95% CI excludes 0.5)"
        if lo >= 0.45 and hi <= 0.55:
            msg += (f" — but its magnitude is at most trivial: any monotone use of current "
                    f"confidence has AUC-equivalent <= {bound:.3f}")
        print(msg + ".")
        if hi < 0.5:
            print("Direction: REVERSED — among currently-wrong answers, HIGHER confidence predicts")
            print("that the next step fixes it. The intuitive low-confidence-continue stopping rule")
            print("would be WORSE than random; confidence still fails as a usable stopping signal,")
            print("which supports the paper's practical thesis.")
        else:
            print("Direction: intuitive — lower confidence -> more likely the next step helps.")
    elif lo >= 0.45 and hi <= 0.55:
        # equivalence-style bound: the CI lies entirely inside the trivial band, so this
        # is affirmative evidence of absence, not merely a failure to detect
        print(f"H2 SUPPORTED (equivalence bound): the 95% CI lies within [0.45, 0.55], so any")
        print(f"monotone use of current confidence has AUC-equivalent at most {bound:.3f} —")
        print("at most trivial. On this run, confidence did not indicate whether another step would help.")
    else:
        print("Consistent with H2 but INCONCLUSIVE: the CI spans 0.5 and is too wide to bound")
        print("the signal as trivial. Raise N_EXAMPLES for a tighter interval.")

# secondary: token-logprob confidence (paper sec. 11A), if this cache captured it
_lps = [s.get("logprob_mean") for t in trajs for s in t["steps"]]
if sum(v is not None for v in _lps) > len(_lps) * 0.5:
    y2, s2 = [], []
    for t in trajs:
        st = t["steps"]
        for i in range(len(st)-1):
            if st[i]["correct"] == 0 and st[i].get("logprob_mean") is not None:
                y2.append(st[i+1]["correct"]); s2.append(-st[i]["logprob_mean"])
    if len(set(y2)) > 1:
        print(f"\n[secondary, descriptive] TOKEN-LOGPROB confidence (mean generated-token logprob),")
        print(f"conditioned improvement AUC = {safe_auc(np.array(y2), np.array(s2)):.3f} — tests whether a")
        print("non-verbal uncertainty signal does what stated confidence cannot (paper sec. 11A).")
else:
    print("\n[note] token logprobs are not in this cache; runs generated with CAPTURE_LOGPROBS=True")
    print("(and an endpoint that supports it) also test the paper's sec. 11A token-probability signal.")

# saturation check (context for interpreting every confidence result above)
allconf = np.array([s["confidence"] for t in trajs for s in t["steps"]])
sat = float((allconf >= 95).mean())
print(f"\n[context] {sat:.0%} of all reported confidences are >= 95. If this is high, the")
print("confidence signal is SATURATED — it cannot discriminate anything, which is itself a")
print("reportable finding about verbal confidence as a stopping signal.")'''

H3_MD = r'''# H3 — Transition-Aware Signals Beat Confidence Alone
**Claim:** signals describing the *refinement process* — did the answer change recently?
is it oscillating? did the critique flag a concrete error? how did confidence move? —
predict the value of another iteration better than confidence alone.

**How we test it:** 13 per-step features available *before* deciding to continue —
including features **mined from the stored critique/solution text** (critique length,
whether it cites a concrete number, solution-length growth) — grouped into the paper's
sec. 11 signal families (A confidence, B stability, C critique, D history/context).
We report each family's solo AUC plus all-features, for **both** next-step improvement
and next-step harm, with a separate verdict for each. Logistic regression,
**group-stratified cross-validation** (questions never straddle folds).
Two methodology guards worth knowing about:
1. Out-of-fold scores are **rank-normalized within each fold** before computing AUC.
   Pooling raw probabilities from different folds is systematically biased toward AUC ≈ 0
   when positive events are rare (the fold holding the positives trains on a lower base
   rate and scores its whole test split lower) — we verified this empirically.
2. The verdict comes from a **paired question-level bootstrap** of the AUC difference
   (transition-aware − confidence-only), gated by a minimum-event guard.

> Upload the `trajectories.json` produced by the H1 notebook before running.'''

H3_CODE = r'''from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
trajs = get_trajectories()
if not any(s.get("critique_text") for t in trajs for s in t["steps"][1:]):
    print("*** WARNING: this cache predates critique_text/full_text storage. The 3 text-mined")
    print("*** features are constant zeros here, so family C/D rows below carry NO text signal.")
    print("*** Regenerate the cache to test the text-mined claims.")

# 13 features, grouped into the paper's sec. 11 signal families. Text features are mined
# from the stored critique/solution text at zero extra API cost. cur["critique_error"] /
# critique_text belong to the critique that PRODUCED the current answer — known at
# decision time, so nothing here is leaky.
FEATS = ["conf", "conf_delta", "changed_now", "changes_recent3", "oscillating",
         "iter_frac", "critique_flagged", "answer_diversity", "unchanged_streak",
         "critique_len", "critique_cites_number", "solution_growth", "is_math"]
FAMILIES = {
    "A:confidence": [0, 1],
    "B:stability":  [2, 3, 4, 7, 8],
    "C:critique":   [6, 9, 10],
    "D:history":    [5, 11, 12],
}

def step_features(t, i):
    st = t["steps"]; cur = st[i]
    conf = cur["confidence"]/100.0
    conf_prev = st[i-1]["confidence"]/100.0 if i > 0 else conf
    n_changes = sum(s["changed"] for s in st[max(0, i-2):i+1])
    cs = [s["canon"] for s in st[:i+1]]
    osc = int(len(cs) >= 3 and cs[-1] == cs[-3] and cs[-1] != cs[-2])
    streak = 0
    for s in reversed(st[1:i+1]):
        if s["changed"] == 0: streak += 1
        else: break
    crit = cur.get("critique_text") or ""
    sol = cur.get("full_text") or ""
    sol_prev = (st[i-1].get("full_text") or "") if i > 0 else sol
    return [conf, conf - conf_prev, cur["changed"], n_changes, osc, i/MAX_ITERS,
            cur["critique_error"], len(set(cs))/(i + 1.0), streak/MAX_ITERS,
            min(len(crit), 2000)/2000.0, int(bool(re.search(r"\d", crit))),
            (len(sol) - len(sol_prev))/1000.0, int(t["kind"] == "math")]

X_full, Yimp, Yharm, Q, KIND = [], [], [], [], []
for qid, t in enumerate(trajs):
    st = t["steps"]
    for i in range(len(st)-1):
        X_full.append(step_features(t, i))
        Yimp.append(int(st[i]["correct"]==0 and st[i+1]["correct"]==1))
        Yharm.append(int(st[i]["correct"]==1 and st[i+1]["correct"]==0))
        Q.append(qid); KIND.append(t["kind"])
X_full = np.array(X_full)
X_conf = X_full[:, FAMILIES["A:confidence"]]
Yimp = np.array(Yimp); Yharm = np.array(Yharm); Q = np.array(Q); KIND = np.array(KIND)
n_imp, n_harm = int(Yimp.sum()), int(Yharm.sum())
print(f"decision points: {len(Q)} | improvement events: {n_imp} | harm events: {n_harm}")

def oof_rank_scores(X, y, n_splits=4, seed=0):
    """Out-of-fold scores, rank-normalized WITHIN each fold. Pooling raw probabilities
    across folds biases AUC toward 0 with rare positives; within-fold ranks are
    calibration-free and immune to that artifact."""
    scores = np.full(len(y), np.nan)
    pos_q = set(Q[y == 1].tolist())
    if len(pos_q) < 2:
        return scores
    skf = StratifiedGroupKFold(n_splits=int(min(n_splits, len(pos_q))),
                               shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y, groups=Q):
        if len(set(y[tr].tolist())) < 2:
            continue
        p = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        scores[te] = rankdata(p) / (len(p) + 1.0)
    return scores

def masked_auc(y, s):
    m = ~np.isnan(s)
    return roc_auc_score(y[m], s[m]) if len(set(y[m].tolist())) > 1 else float("nan")

if min(n_imp, n_harm) < MIN_EVENTS:
    print(f"\nVERDICT: UNTESTABLE on this run — fewer than MIN_EVENTS={MIN_EVENTS} transition")
    print("events. Cross-validated AUCs at single-digit event counts are noise. Raise")
    print("N_EXAMPLES and keep USE_HF_MATH500=True, then regenerate trajectories.json.")
else:
    # signal-family ablation (paper sec. 11): each family alone, then everything together
    S = {}
    for fam, cols in FAMILIES.items():
        S[(fam, "imp")]  = oof_rank_scores(X_full[:, cols], Yimp)
        S[(fam, "harm")] = oof_rank_scores(X_full[:, cols], Yharm)
    S[("ALL", "imp")]  = oof_rank_scores(X_full, Yimp)
    S[("ALL", "harm")] = oof_rank_scores(X_full, Yharm)

    print("\nGroup-stratified CV, within-fold rank-normalized out-of-fold scores:")
    print("(table values are descriptive — only the bootstrap verdicts below are inferential)")
    print(f"{'signal family':>16} {'AUC improve':>12} {'AUC harm':>10}")
    for fam in list(FAMILIES) + ["ALL"]:
        print(f"{fam:>16} {masked_auc(Yimp, S[(fam,'imp')]):>12.3f} "
              f"{masked_auc(Yharm, S[(fam,'harm')]):>10.3f}")
    print("   (caveat: family solo AUCs share an implicit iteration-index proxy — several")
    print("   features correlate with the iteration number — so compare families with caution)")
    for k in sorted(set(KIND.tolist())):
        m = KIND == k
        sc = S[("ALL", "imp")][m]
        mm = ~np.isnan(sc)
        a = (roc_auc_score(Yimp[m][mm], sc[mm])
             if len(set(Yimp[m][mm].tolist())) > 1 else float("nan"))
        print(f"   [descriptive] ALL-features improve AUC on {k} only: {a:.3f}")

    def delta_auc(qs, sa, sb, y):
        rows = np.concatenate([np.where(Q == q)[0] for q in qs])
        yy, va, vb = y[rows], sa[rows], sb[rows]
        m = ~np.isnan(va) & ~np.isnan(vb)
        if len(set(yy[m].tolist())) < 2:
            return float("nan")
        return roc_auc_score(yy[m], vb[m]) - roc_auc_score(yy[m], va[m])

    qids = np.unique(Q); rng = np.random.default_rng(0)
    for target, y in (("improvement", Yimp), ("harm", Yharm)):
        key = "imp" if target == "improvement" else "harm"
        deltas = []
        for _ in range(1000):
            v = delta_auc(qids[rng.integers(0, len(qids), len(qids))],
                          S[("A:confidence", key)], S[("ALL", key)], y)
            if v == v:
                deltas.append(v)
        point = delta_auc(qids, S[("A:confidence", key)], S[("ALL", key)], y)
        if deltas:
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            print(f"\npaired AUC delta (ALL - confidence-only) on {target}: {point:+.3f}")
            print(verdict(f"H3 on {target} (transition signals beat confidence)",
                          float(lo), float(hi)))
        else:
            print(f"\nH3 on {target}: UNTESTABLE — every bootstrap resample was single-class.")

print("\nfeature families (paper sec. 11): A confidence, B answer-stability,")
print("C critique quality (incl. text-mined), D history/context (incl. text-mined)")'''

H4_MD = r'''# H4 — Selective Refinement Is More Efficient
**Claim:** a learned "is the next step worth it?" predictor achieves a better
**accuracy vs cost** trade-off than fixed iteration counts, confidence thresholds,
answer-stability stopping, or an external judge alone.

**How we test it:** replay each trajectory under every stopping policy and report the
paper's metric table (sec. 13): final accuracy, average iterations, **token cost**,
harmful-flip rate, **successful-correction rate**, and **regret vs oracle**.

**Two learned policies plus a hybrid rule** (value predictions are always out-of-fold via
group-stratified CV, calibrated per fold by subtracting the fold's training base rate and
re-anchored at the global base rate — never predicted on their own training questions).
Both learned policies use the **same 13 features** — including three **mined from the
stored critique/solution text** (critique length, whether it cites a concrete number,
solution-length growth — paper sec. 11C/11D) plus answer diversity, unchanged-streak, and
task type — and differ **only** in the estimator and the label horizon. Zero extra API
cost: everything replays from the cached trajectories.
- `learned` — logistic regression on a **myopic** one-step value
  (P(next step fixes) − P(next step breaks)).
- `learned+` — small gradient-boosted trees on a **continue-to-end** value
  (P(final answer better than current) − P(worse)); this label assumes the
  always-continue behavior policy (standard off-policy approximation).
- `hybrid` — stop at the first of (answer unchanged) OR (learned+ predicted value ≤ 0).

Honesty note: these policies and hyperparameters were developed against this cached run;
confirmation on fresh trajectories (e.g. the `PROMPT_STYLE = "conservative"` rerun) is the
convincing test.

Besides the single τ = 0 operating point, we sweep the decision threshold to trace the
learned policy's whole **accuracy–cost frontier** (the paper's sec. 9 operating
settings: accuracy-focused / cost-aware / safety-focused are different τ). The τ = 0
point is the a-priori headline (no post-hoc selection); frontier points at other τ are
descriptive, and the matched-cost comparison against stability is labeled as post-hoc.

Baselines include an **external-judge** policy (stop when the reviewer critique says
"no error found"; replays for free since critiques are stored, though its true cost is
~one extra critique beyond the stopping step, and it is a same-model judge — a proxy
for the paper's "another model"). Oracle = best possible stop, upper bound only.

The final cell tests whether the predictor **transfers between datasets** (train on math,
evaluate on QA, and vice versa — paper sec. 19, analysis 6): a pure replay, no API cost.

> Reads the same `trajectories.json` (Google Drive) produced by the H1 notebook.'''

H4_CODE = r'''from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
trajs = get_trajectories()
if not any(s.get("critique_text") for t in trajs for s in t["steps"][1:]):
    print("*** WARNING: this cache predates critique_text/full_text storage. The 3 text-mined")
    print("*** features are constant zeros here, so learned/learned+ (and the transfer cell)")
    print("*** carry NO text signal. Regenerate the cache to test the text-mined claims.")

FEATS = ["conf", "conf_delta", "changed_now", "changes_recent3", "oscillating",
         "iter_frac", "critique_flagged", "answer_diversity", "unchanged_streak",
         "critique_len", "critique_cites_number", "solution_growth", "is_math"]

def step_features(t, i):
    """Everything here is known at decision time; the text features are mined from the
    stored critique/solution text (paper sec. 11C/11D) at zero extra API cost."""
    st = t["steps"]; cur = st[i]
    conf = cur["confidence"]/100.0
    conf_prev = st[i-1]["confidence"]/100.0 if i > 0 else conf
    n_changes = sum(s["changed"] for s in st[max(0, i-2):i+1])
    cs = [s["canon"] for s in st[:i+1]]
    osc = int(len(cs) >= 3 and cs[-1] == cs[-3] and cs[-1] != cs[-2])
    streak = 0
    for s in reversed(st[1:i+1]):
        if s["changed"] == 0: streak += 1
        else: break
    crit = cur.get("critique_text") or ""
    sol = cur.get("full_text") or ""
    sol_prev = (st[i-1].get("full_text") or "") if i > 0 else sol
    return [conf, conf - conf_prev, cur["changed"], n_changes, osc, i/MAX_ITERS,
            cur["critique_error"], len(set(cs))/(i + 1.0), streak/MAX_ITERS,
            min(len(crit), 2000)/2000.0, int(bool(re.search(r"\d", crit))),
            (len(sol) - len(sol_prev))/1000.0, int(t["kind"] == "math")]

X, Vimp, Vharm, Vbet, Vwor, Q = [], [], [], [], [], []
for qid, t in enumerate(trajs):
    st = t["steps"]
    for i in range(len(st)-1):
        X.append(step_features(t, i))
        Vimp.append(int(st[i]["correct"]==0 and st[i+1]["correct"]==1))
        Vharm.append(int(st[i]["correct"]==1 and st[i+1]["correct"]==0))
        # continue-to-end labels (always-continue behavior policy, the standard
        # off-policy approximation): does running out the full loop end better/worse
        # than stopping here?
        Vbet.append(int(st[i]["correct"]==0 and st[-1]["correct"]==1))
        Vwor.append(int(st[i]["correct"]==1 and st[-1]["correct"]==0))
        Q.append(qid)
X = np.array(X); Vimp = np.array(Vimp); Vharm = np.array(Vharm)
Vbet = np.array(Vbet); Vwor = np.array(Vwor); Q = np.array(Q)
n_imp, n_harm = int(Vimp.sum()), int(Vharm.sum())
print(f"decision points: {len(Q)} | improvement events: {n_imp} | harm events: {n_harm} | "
      f"end-better: {int(Vbet.sum())} | end-worse: {int(Vwor.sum())}")
if min(n_imp, n_harm) < MIN_EVENTS:
    print(f"WARNING: fewer than MIN_EVENTS={MIN_EVENTS} transition events — the learned rows")
    print("below are trained on almost nothing and their comparisons are NOT meaningful.")

def oof_calibrated(make_est, Xa, y, n_splits=4, seed=0):
    """Out-of-fold P(y=1) minus each fold's training base rate. Subtracting the base rate
    removes the fold-calibration artifact that biases pooled cross-fold probabilities."""
    out = np.zeros(len(y))
    pos_q = set(Q[y == 1].tolist())
    if len(pos_q) < 2:
        return out                              # nothing learnable — contribute zero value
    skf = StratifiedGroupKFold(n_splits=int(min(n_splits, len(pos_q))),
                               shuffle=True, random_state=seed)
    for tr, te in skf.split(Xa, y, groups=Q):
        if len(set(y[tr].tolist())) < 2:
            out[te] = 0.0
            continue
        m = make_est().fit(Xa[tr], y[tr])
        out[te] = m.predict_proba(Xa[te])[:, 1] - y[tr].mean()
    return out

LR  = lambda: LogisticRegression(max_iter=1000)
# GBM is deliberately small: sklearn's automatic early stopping is OFF below 10k samples,
# so an uncapped model runs all its rounds and overfits ~1.3k-row training folds badly
# (verified: train AUC ~0.9 vs OOF ~0.5), making the stop/continue decision noise-driven.
GBM = lambda: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.1,
                                             max_iter=40, min_samples_leaf=50,
                                             random_state=0)

# expected value of continuing = P(better) - P(worse); each probability is out-of-fold,
# fold-recalibrated, then re-anchored at the GLOBAL base rate so it is an absolute
# estimate — without re-anchoring, "value > 0" would mean "better than the average step"
# and the policy would collapse to fixed@0 whenever per-item signal is weak.
val_myo = (oof_calibrated(LR, X, Vimp) + Vimp.mean()) - (oof_calibrated(LR, X, Vharm) + Vharm.mean())
val_end = (oof_calibrated(GBM, X, Vbet) + Vbet.mean()) - (oof_calibrated(GBM, X, Vwor) + Vwor.mean())

def _auc(y, s):
    return roc_auc_score(y, s) if len(set(y.tolist())) > 1 else float("nan")
print(f"value-model OOF AUCs (descriptive): myopic improve {_auc(Vimp, val_myo):.3f} | "
      f"end-better {_auc(Vbet, val_end):.3f} | end-worse {_auc(Vwor, -val_end):.3f}")

pv_myo, pv_end = {}, {}
row = 0
for qid, t in enumerate(trajs):
    for i in range(len(t["steps"]) - 1):
        pv_myo[(qid, i)] = float(val_myo[row]); pv_end[(qid, i)] = float(val_end[row]); row += 1
assert row == len(X), "value predictions misaligned with decision points"

def evaluate(stop_fn):
    n = len(trajs); corr = iters = toks = harm = fix = 0
    for qid, t in enumerate(trajs):
        st = t["steps"]; s = stop_fn(qid, st)
        corr  += st[s]["correct"]
        iters += s
        toks  += sum(x["tokens"] for x in st[:s+1])
        # harmful flip: returned wrong although an earlier step had it right
        harm  += int(st[s]["correct"] == 0 and any(x["correct"] for x in st[:s]))
        # successful correction: initially wrong, returned right
        fix   += int(st[s]["correct"] == 1 and st[0]["correct"] == 0)
    return {"acc": corr/n, "iters": iters/n, "tokens": toks/n, "harm": harm/n, "fix": fix/n}

policies = {}
for kk in range(MAX_ITERS + 1):
    policies[f"fixed@{kk}"] = (lambda k2: (lambda qid, st: k2))(kk)
def conf_thresh(tau):
    def f(qid, st):
        for i, s in enumerate(st):
            if s["confidence"] >= tau:
                return i
        return len(st) - 1
    return f
policies["conf>=80"] = conf_thresh(80)
policies["conf>=90"] = conf_thresh(90)
def stability(qid, st):                        # canonical-form comparison (not raw strings)
    for i in range(1, len(st)):
        if st[i]["changed"] == 0:
            return i
    return len(st) - 1
policies["stability"] = stability
def judge(qid, st):
    # external-judge baseline (paper sec. 12): stop once the reviewer critique of the
    # CURRENT answer finds no error. st[i+1]["critique_error"] is that verdict (the
    # critique that produced step i+1 reviewed answer i), so this replays at zero API
    # cost; its true cost is ~one critique beyond the stop, slightly above avg_iters.
    for i in range(len(st) - 1):
        if st[i+1]["critique_error"] == 0:
            return i
    return len(st) - 1
policies["judge"] = judge
def learned(qid, st):
    for i in range(len(st) - 1):
        if pv_myo[(qid, i)] <= 0.0:              # stop when the next step isn't worth it
            return i
    return len(st) - 1
policies["learned"] = learned
def learned_plus(qid, st):
    for i in range(len(st) - 1):
        if pv_end[(qid, i)] <= 0.0:              # stop when continuing isn't worth it
            return i
    return len(st) - 1
policies["learned+"] = learned_plus
def hybrid(qid, st):
    # stop at the FIRST of: answer stabilized, or predicted continue-value <= 0
    for i in range(len(st) - 1):
        if (i >= 1 and st[i]["changed"] == 0) or pv_end[(qid, i)] <= 0.0:
            return i
    return len(st) - 1
policies["hybrid"] = hybrid
def oracle(qid, st):
    good = [i for i, s in enumerate(st) if s["correct"] == 1]
    return min(good) if good else 0
policies["oracle"] = oracle

res = {name: evaluate(fn) for name, fn in policies.items()}
oracle_acc = res["oracle"]["acc"]
print("\n(policy table is descriptive — no CIs; only the bootstrap verdicts below are inferential)")
print(f"{'policy':>10} {'accuracy':>9} {'avg_iters':>10} {'avg_tokens':>11} "
      f"{'harm_rate':>10} {'fix_rate':>9} {'regret':>8}")
for name, m in res.items():
    print(f"{name:>10} {m['acc']:>9.3f} {m['iters']:>10.2f} {m['tokens']:>11.0f} "
          f"{m['harm']:>10.3f} {m['fix']:>9.3f} {oracle_acc - m['acc']:>8.3f}")

# ---- accuracy-cost FRONTIER of learned+: sweep the stopping threshold tau ----
# (tau = 0 is the a-priori operating point; other points are descriptive — choosing a
# tau after seeing this plot is post-hoc selection and is labeled as such below)
if val_end.max() - val_end.min() < 1e-12:
    print("[note] the value model output is constant (no learnable continue-to-end signal);")
    print("the 'frontier' below is a single degenerate point, not a curve")
taus = np.unique(np.quantile(val_end, np.linspace(0.02, 0.98, 25)))
frontier = []
for tau in taus:
    def pol(qid, st, tau=tau):
        for i in range(len(st) - 1):
            if pv_end[(qid, i)] <= tau:
                return i
        return len(st) - 1
    m = evaluate(pol)
    frontier.append((m["iters"], m["acc"]))
frontier.sort()

plt.figure(figsize=(10, 6))
fx = [p[0] for p in frontier]; fy = [p[1] for p in frontier]
plt.plot(fx, fy, "-", lw=2, alpha=0.5, color="tab:purple",
         label="learned+ frontier (tau sweep,\ndescriptive, in-sample)")

# fixed@k as one gray series with small k digits (they are spread out, so digits stay clear)
fixed_pts = [(res[f"fixed@{k}"]["iters"], res[f"fixed@{k}"]["acc"], k) for k in range(MAX_ITERS + 1)]
plt.plot([p[0] for p in fixed_pts], [p[1] for p in fixed_pts], ":", color="gray", lw=1, alpha=0.7)
plt.scatter([p[0] for p in fixed_pts], [p[1] for p in fixed_pts], s=45, color="gray",
            zorder=3, label="fixed@k (digit = k)")
for x, y, k in fixed_pts:
    plt.annotate(str(k), (x, y), textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=9, color="dimgray")

# every other policy: distinct marker + color, identified in the legend (no inline text).
# conf>=80/90 coincide with fixed@0, so they get open markers to stay visible when stacked.
STYLE = {"conf>=80":  ("v", "tab:red",    80, False),
         "conf>=90":  ("^", "tab:orange", 80, False),
         "stability": ("s", "tab:blue",   90, True),
         "judge":     ("P", "tab:brown",  90, True),
         "learned":   ("D", "tab:green",  95, True),
         "learned+":  ("D", "tab:purple", 120, True),
         "hybrid":    ("d", "tab:cyan",   95, True),
         "oracle":    ("*", "goldenrod",  240, True)}
for name, (mk, col, sz, filled) in STYLE.items():
    m = res[name]
    if filled:
        plt.scatter(m["iters"], m["acc"], marker=mk, s=sz, color=col, zorder=4, label=name)
    else:
        plt.scatter(m["iters"], m["acc"], marker=mk, s=sz, facecolors="none",
                    edgecolors=col, linewidths=1.8, zorder=4, label=name)
plt.xlabel("average iterations (cost)", fontsize=11); plt.ylabel("final accuracy", fontsize=11)
plt.title("H4: accuracy vs cost (tau=0 is pre-specified; swept taus are post-hoc)", fontsize=12)
plt.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10)
plt.grid(alpha=.3)
plt.tight_layout(); plt.show()

# matched-cost comparison vs the strongest cheap baseline (POST-HOC tau selection —
# report as descriptive, not as the headline test)
stab = res["stability"]
at_cost = [p for p in frontier if p[0] <= stab["iters"] + 1e-9]
if at_cost:
    best = max(p[1] for p in at_cost)
    print(f"[descriptive, post-hoc] learned+ best-of-swept-tau at stability-matched cost "
          f"(<= {stab['iters']:.2f} iters): accuracy {best:.3f} vs stability {stab['acc']:.3f} — tau was"
          f" chosen on this same data (max over the sweep, no CI), so this is an optimistic upper"
          f" bound; the CI-gated verdicts below, not this line, are the headline result.")

if min(n_imp, n_harm) < MIN_EVENTS:
    print("\nH4 VERDICT: UNTESTABLE on this run — too few transition events to train or judge")
    print("the learned policies (see the WARNING above).")
else:
    BASE = [p for p in policies if p not in ("learned", "learned+", "hybrid", "oracle")]
    def h4_stat(sample_qids, pol_name):
        def replay(fn):
            c = it = 0
            for qid in sample_qids:
                st = trajs[qid]["steps"]; s = fn(qid, st)
                c += st[s]["correct"]; it += s
            return c/len(sample_qids), it/len(sample_qids)
        a_l, it_l = replay(policies[pol_name])
        accs = [replay(policies[nm])[0] for nm in BASE
                if replay(policies[nm])[1] <= it_l + 1e-9]
        return (a_l - max(accs)) if accs else float("nan")
    qids = np.arange(len(trajs)); rng = np.random.default_rng(0)
    print("\nThree policy variants are tested below — three separate claims, so read the")
    print("verdicts jointly: at the 95% level, one isolated SUPPORTED among three tests is")
    print("weaker evidence than three consistent ones. learned+ (continue-to-end) is the")
    print("primary test; learned and hybrid are secondary.")
    for pol_name in ("learned", "learned+", "hybrid"):
        deltas = []
        for _ in range(1000):
            v = h4_stat(qids[rng.integers(0, len(qids), len(qids))], pol_name)
            if v == v:
                deltas.append(v)
        if deltas:
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            print(f"\n{pol_name} - best equal-or-cheaper baseline accuracy: "
                  f"{h4_stat(qids, pol_name):+.3f}")
            print(verdict(f"H4 ({pol_name} beats equal-or-cheaper baselines)", float(lo), float(hi)))
        else:
            print(f"\nH4 ({pol_name}): UNTESTABLE — bootstrap could not form valid comparisons.")'''

H4_TRANSFER = r'''# ---- H4 extra: cross-dataset TRANSFER of the value predictor (paper sec. 19, analysis 6) ----
from sklearn.metrics import roc_auc_score
KIND = np.array([t["kind"] for t in trajs for _ in range(len(t["steps"]) - 1)])

def transfer_auc(a, b, y):
    tr, te = KIND == a, KIND == b
    if tr.sum() == 0 or te.sum() == 0 \
       or len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
        return float("nan")
    m = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
    return roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])

print("Cross-dataset TRANSFER (train the value predictor on one dataset, test on the other)")
print("[descriptive: single split, no CI — treat as directional only]:")
for a, b in [("math", "qa"), ("qa", "math")]:
    print(f"  train {a:>4} -> test {b:<4}  AUC(improve)={transfer_auc(a, b, Vimp):.3f}  "
          f"AUC(harm)={transfer_auc(a, b, Vharm):.3f}")
print(f"\nrows: math={int((KIND=='math').sum())}, qa={int((KIND=='qa').sum())}")
print("nan = one side has no data or no events of that class. The default run uses MATH-500")
print("only; enable USE_HF_HOTPOT=True as well (and raise N_EXAMPLES) to fill the qa side.")
print("Failure to transfer is itself a reportable result (paper sec. 18).")'''

HOWTO = r'''## How to run (read me first)

**Easiest path: use `ALL_hypotheses_full_study.ipynb`** — every section in one notebook,
run top to bottom. The four individual notebooks contain the same code for modular reruns.

1. Open this notebook in **Google Colab** (File ▸ Upload notebook).
2. Get a **free NVIDIA API key**: go to https://build.nvidia.com , open any model page,
   click **Get API Key**. It looks like `nvapi-...`.
3. Run the **SHARED HARNESS** cell — it asks for the key and **validates the key and model
   id immediately** (clear error message if either is wrong).
4. Run the hypothesis cell(s) below.

**Order matters for cost:** run **H1 first** — it generates `trajectories.json`
(each question through the full critique→revise loop: 9 API calls per question;
446 questions ≈ 4000 calls ≈ 2–2.5 h on the free tier, with progress + ETA printed).

**Crash-proof by default:** the harness mounts your **Google Drive** (authorize the popup
once per session) and saves the cache to
`MyDrive/refinement_experiments/trajectories_<style>.json` after **every question** — a
runtime crash or disconnect loses nothing; just re-run the cells and it continues. The
filename includes the prompt style, so a `"conservative"` ablation run can never overwrite
your `"neutral"` run (an older `trajectories.json` is adopted as the neutral cache
automatically). Resume matches by **question text**, so raising `N_EXAMPLES` later or
starting from an older checkpoint is always safe: already-run questions are kept, only new
ones are paid for. All notebooks read the same Drive file — **no downloading/uploading
between notebooks**. (A notebook only regenerates if the file is missing, empty, or was
built with a different `MODEL` / `MAX_ITERS` / `PROMPT_STYLE` — the cache is stamped.)

**Token logprobs:** new generation runs also record each response's mean token
log-probability (`CAPTURE_LOGPROBS = True`) — the paper's sec. 11A "token-level
probability" signal — so H2 can test a non-verbal confidence too. Caches generated before
this feature simply skip that secondary analysis.

**Data:** defaults to **MATH-500** (level ≥ 3, plain-numeric answers; 223 items) plus
**HotpotQA** multi-hop questions with their context paragraphs (223 items) — hard enough
that the model is wrong a useful fraction of the time, which is what makes the four
hypotheses measurable, and two datasets enable the cross-dataset transfer test. If the
download fails you get a built-in 20-question smoke-test set: fine for checking the
plumbing, far too easy for real results (expect UNTESTABLE/INCONCLUSIVE verdicts on it).

**Verdicts:** every notebook ends with `SUPPORTED` / `NOT SUPPORTED` / `INCONCLUSIVE` /
`UNTESTABLE`, based on question-level bootstrap confidence intervals and minimum-event
guards — never a bare point estimate. Each step also stores the model's `full_text` and
`critique_text` in `trajectories.json`, so any scoring decision can be audited later
without re-running the API.

**Knobs** (top of the harness): `MODEL` (Nemotron 3 family ids listed), `MAX_ITERS`,
`N_EXAMPLES`, `PROMPT_STYLE` (`"neutral"` vs `"conservative"` revision instruction —
paper sec. 18 asks to compare both), `USE_HF_MATH500` / `USE_HF_HOTPOT`.'''


def cell_md(src):   return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
def cell_code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                            "outputs": [], "source": src.splitlines(keepends=True)}

def notebook(title_md, harness, body_cells):
    cells = [cell_md(title_md), cell_md(HOWTO), cell_code(harness)] + body_cells
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"},
                         "colab": {"provenance": []}},
            "nbformat": 4, "nbformat_minor": 5}

COMBINED_TITLE = r'''# Confidence Is Not a Stopping Rule — Full Study (H1–H4)
All four hypothesis experiments in one notebook, run **top to bottom**:
the shared harness runs once, generation happens once (or loads the Google-Drive cache),
and each hypothesis section replays the same trajectories with its own analysis.

| Section | Hypothesis | Headline output |
|---|---|---|
| H1 | Confidence inflation | accuracy/confidence curves, transition table with net-value CIs, saturation diagnosis |
| H2 | Current confidence is not enough | conditioned AUCs, equivalence bound, token-logprob check |
| H3 | Transition-aware signals beat confidence | signal-family ablation (paper sec. 11 A–D), paired-delta verdicts |
| H4 | Selective refinement is more efficient | policy table, accuracy-cost frontier, transfer test |

Every verdict line is gated by question-level bootstrap CIs and minimum-event guards.'''

RESULTS_MD = r'''---
# Results — draft write-up (from the 446-question neutral-prompt run)

*Numbers below are from the completed run of this notebook (nvidia/nemotron-3-nano-30b-a3b,
223 MATH-500 level≥3 numeric + 223 HotpotQA questions with context, 4 refinement
iterations, neutral revision prompt; 1,784 decision points, 153 wrong→right and 113
right→wrong transitions; 16/2,230 confidence parses failed, 0.7%). If you regenerate the
data, update the numbers from the cell outputs above.*

## R1. Confidence is saturated, not merely inflated (H1)

Stated confidence starts at ceiling and stays there: iteration-0 mean confidence is
98/100 while iteration-0 accuracy is 0.41, and **98% of all reported confidences are
≥ 95** across all iterations. The hypothesized *inflation dynamic* (confidence rising
faster than accuracy) is therefore untestable by construction — there is no room left to
rise — and the run-level statistic is significantly negative
((conf rise) − (acc rise) = −0.075, 95% question-bootstrap CI [−0.119, −0.028]), driven
by accuracy improving under refinement while confidence stays pinned. The reportable
finding is the **constant overconfidence gap of ~0.5** itself.

Transition-level analysis (the paper's four-way protocol) shows refinement's value is
front-loaded: only the **first** revision has significantly positive net value
(+0.065 per question, CI [+0.027, +0.105]; 54 fixes vs 25 harmful flips), while rounds
2–4 are indistinguishable from zero (+0.016 [−0.020, +0.052]; +0.011 [−0.018, +0.045];
−0.002 [−0.038, +0.034] — by round 4, 35 fixes vs 36 flips). Accuracy peaks at 0.50 by
iteration 3. Per dataset, the negative pooled statistic is driven by math
(−0.194 [−0.262, −0.122]); QA shows a non-significant hint of true inflation
(+0.043 [−0.010, +0.103]).

## R2. Current-answer confidence carries no usable next-step signal (H2)

Conditioned on the current answer being wrong (so that current-correctness signal cannot
leak into the test — the paper's Q1/Q2 distinction), stated confidence does not predict
which answers the next revision will fix: **AUC = 0.485, 95% CI [0.466, 0.506]**
(n = 950 conditioned decision points, 153 improvement events). Because the CI lies
entirely inside the [0.45, 0.55] triviality band, this is an **equivalence-style bound**,
not merely a failure to detect: any monotone use of stated confidence has AUC-equivalent
at most 0.534. Symmetrically, among currently-correct answers confidence does not predict
harmful flips (AUC = 0.495). Mechanically, 1,764 of 1,784 decision points share one
confidence band (≥ 80), so confidence-threshold stopping rules degenerate (see R4).

## R3. Process signals predict next-step value where confidence cannot (H3)

Thirteen decision-time features grouped into the paper's §11 signal families, evaluated
with group-stratified CV and within-fold rank normalization:

| family | AUC improve | AUC harm |
|---|---|---|
| A: confidence | 0.530 | 0.454 |
| B: answer stability | 0.635 | 0.578 |
| C: critique quality (incl. text-mined) | 0.549 | 0.571 |
| D: history/context (incl. text-mined) | 0.639 | 0.485 |
| **ALL** | **0.667** | **0.551** |

Both headline comparisons are significant: transition-aware features beat
confidence-only on predicting improvement (ΔAUC = +0.137, CI [+0.080, +0.193]) **and** on
predicting harmful flips (ΔAUC = +0.097, CI [+0.025, +0.167]). Stability and history
signals carry the improvement signal; critique quality is the best harm predictor;
confidence contributes essentially nothing. Signals are more informative on QA
(improve AUC 0.723) than math (0.570). All features replay from stored trajectories at
zero additional inference cost.

## R4. Value-aware stopping: best point estimates, honest inconclusive verdicts (H4)

Replaying all stopping policies on the same trajectories: confidence thresholds are
**vacuous** under saturation (conf ≥ 80/90 stop after ~0.04 iterations, accuracy 0.408 —
identical to never refining), and the same-model judge stops too early (accuracy 0.422 at
0.30 iterations — the model's overconfidence extends to its critic role: "no error found"
on wrong answers). The continue-to-end value policy `learned+` achieves the **best
non-oracle accuracy at mid cost (0.509 at 1.95 iterations / 2,693 tokens)**, strictly
dominating fixed@3 (0.500 at 3.0 iterations / 4,873 tokens — 45% more tokens for less
accuracy) and edging answer-stability (0.496 at 1.62). However, against the best
equal-or-cheaper baseline, all three learned-policy comparisons are positive in point
estimate but **inconclusive** at n = 446 (learned +0.004 [−0.022, +0.027];
learned+ +0.013 [−0.013, +0.038]; hybrid +0.013 [−0.011, +0.038]). Oracle stopping
reaches 0.659 at 0.50 iterations — a regret of 0.150 for the best policy — quantifying
the headroom that better stopping policies could claim.

## R5. Fixability transfers across tasks; harm risk does not

A value predictor trained only on math transfers to QA for improvement prediction
(AUC 0.656 vs 0.723 within-domain), and QA→math transfers with no degradation at all
(0.570 vs 0.570 within-domain). Harm prediction does **not** transfer in either direction
(0.509 / 0.472 ≈ chance; single-split, directional). The same asymmetry appeared in an
earlier independent run. Implication: a general-purpose "will another step help?" head is
plausible, but the safety side ("might another step hurt?") appears task-specific and
needs per-task calibration — mapping onto the paper's accuracy-focused vs safety-focused
operating modes (§9).

## Limitations

Single model (a 30B-active-parameter MoE) and a single revision-prompt style so far (the
`PROMPT_STYLE="conservative"` ablation and token-logprob capture are wired into this
notebook for a follow-up run); verbal confidence only in this dataset; answer scoring
uses numeric/substring matching (raw model text is stored for audit); the H4 policies
were developed against this cached run, so their inconclusive-positive comparisons await
confirmation on fresh trajectories.
'''

specs = [
 ("H1_confidence_inflation.ipynb",      H1_MD, [cell_code(H1_CODE)]),
 ("H2_confidence_not_enough.ipynb",     H2_MD, [cell_code(H2_CODE)]),
 ("H3_transition_aware_signals.ipynb",  H3_MD, [cell_code(H3_CODE)]),
 ("H4_selective_refinement.ipynb",      H4_MD, [cell_code(H4_CODE), cell_code(H4_TRANSFER)]),
]

if __name__ == "__main__" or True:
    for fname, md, body in specs:
        nb = notebook(md, SHARED, body)
        with open(os.path.join(OUT, fname), "w") as f:
            json.dump(nb, f, indent=1)
        print("wrote", fname)
    combined_cells = [cell_md(COMBINED_TITLE), cell_md(HOWTO), cell_code(SHARED),
                      cell_md(H1_MD), cell_code(H1_CODE),
                      cell_md(H2_MD), cell_code(H2_CODE),
                      cell_md(H3_MD), cell_code(H3_CODE),
                      cell_md(H4_MD), cell_code(H4_CODE), cell_code(H4_TRANSFER),
                      cell_md(RESULTS_MD)]
    nb = {"cells": combined_cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"},
                       "colab": {"provenance": []}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(os.path.join(OUT, "ALL_hypotheses_full_study.ipynb"), "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote ALL_hypotheses_full_study.ipynb")
    print("ALL DONE ->", OUT)
