# Artifact: "Confidence Is Not a Stopping Rule"

Anonymous artifact for the workshop submission.

- `trajectories.json` — all 446 question trajectories (5 steps each: answer, stated
  confidence, correctness, critique text, full solution text, token counts).
  This is the complete dataset every number in the paper is computed from.
- `ALL_hypotheses_full_study.ipynb` — the generation harness and the full analysis
  (Colab-ready; regenerates trajectories from scratch given an API key).
- `build_notebooks.py` — generator for the notebook above.
- `abs_auc.py` — absolute AUC estimates with question-level clustered bootstrap CIs.
- `critic.py` — critic-verdict analysis and the verdict-permutation null.
- `transfer.py` — cross-domain transfer, including the shared-canonicalizer control.

All analysis scripts read `trajectories.json` and require only numpy/scipy/scikit-learn.
