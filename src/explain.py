"""Explanations: permutation importance and a plain-language maintenance memo.
The memo is built from computed numbers only. If GROQ_API_KEY is set, an LLM rewrites it
for a plant manager, but it is instructed to use ONLY the supplied figures."""
import os
import pandas as pd
import requests
from sklearn.inspection import permutation_importance


def top_importances(model, sample, feats, n=15):
    r = permutation_importance(model, sample[feats], sample["rul"], n_repeats=3, random_state=0, n_jobs=1)
    df = pd.DataFrame({"feature": feats, "importance": r.importances_mean})
    return df.sort_values("importance", ascending=False).head(n).reset_index(drop=True)


def facts_text(preds, cost, imp, n=5):
    top = preds.head(n)
    lines = [f"- Engine {int(r.unit)}: predicted {r.pred_rul:.0f} cycles left ({r.band})" for r in top.itertuples()]
    return (f"**Engines assessed:** {cost['n_engines']}\n\n"
            "**Most urgent engines:**\n\n" + "\n".join(lines) +
            f"\n\n**Simulated policy costs:**\n\n"
            f"- Run-to-failure: {cost['run_to_failure']:,.0f}\n"
            f"- Fixed schedule: {cost['fixed_schedule']:,.0f}\n"
            f"- Predictive: {cost['predictive']:,.0f}\n"
            f"- Predictive vs fixed: {cost['saving_vs_fixed_pct']:.1f}%\n"
            f"- Unplanned failures, fixed vs predictive: {cost['failures_fixed']} vs {cost['failures_predictive']}\n\n"
            f"**Signals driving predictions:** {', '.join(imp['feature'].head(3))}")


def llm_memo(facts):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers={"Authorization": f"Bearer {key}"}, timeout=30,
                          json={"model": "llama-3.3-70b-versatile", "temperature": 0.2, "messages": [
                              {"role": "system", "content": "You write short maintenance memos for a plant manager. Use ONLY the numbers given. Never invent figures. State that costs are simulated assumptions."},
                              {"role": "user", "content": facts}]})
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def write_memo(preds, cost, imp, path, synthetic=False):
    facts = facts_text(preds, cost, imp)
    body = llm_memo(facts) or ("Maintenance recommendation (auto-generated from model output)\n\n" + facts +
                               "\n\nNote: cost figures come from a simulation using stated assumptions, not real plant data.")
    warn = "> WARNING: generated from SYNTHETIC data. Not a real result.\n\n" if synthetic else ""
    path.write_text(warn + body)
