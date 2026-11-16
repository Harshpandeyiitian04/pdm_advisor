# This file explains why the model made its guesses, and writes a short maintenance note a manager can read.
# The note is built only from numbers the program already calculated.
# If a Groq key is set in the environment, a language model may rewrite that note, but it is told to use only those numbers.

# These lines bring in the environment, data tables, web calls, and a tool that checks which clues matter most.
import os
import pandas as pd
import requests
from sklearn.inspection import permutation_importance


# This finds the clues that change the guess the most when they are scrambled.
def top_importances(model, sample, feats, n=15):
    # Scramble each clue a few times and see how much the guesses get worse.
    r = permutation_importance(model, sample[feats], sample["rul"], n_repeats=3, random_state=0, n_jobs=1)
    # Pair each clue with how much it mattered.
    df = pd.DataFrame({"feature": feats, "importance": r.importances_mean})
    # Keep the strongest clues, biggest first.
    return df.sort_values("importance", ascending=False).head(n).reset_index(drop=True)


# This turns the results into a short plain-text list: the most urgent engines, the costs, and the top clues.
def facts_text(preds, cost, imp, n=5):
    # Take the engines that look closest to needing work.
    top = preds.head(n)
    # One line per urgent engine: how much life is left and how urgent it is.
    lines = [f"- Engine {int(r.unit)}: predicted {r.pred_rul:.0f} cycles left ({r.band})" for r in top.itertuples()]
    # Join the engine count, the urgent engines, the three cost plans, and the top clues into one note.
    return (f"**Engines assessed:** {cost['n_engines']}\n\n"
            "**Most urgent engines:**\n\n" + "\n".join(lines) +
            f"\n\n**Simulated policy costs:**\n\n"
            f"- Run-to-failure: {cost['run_to_failure']:,.0f}\n"
            f"- Fixed schedule: {cost['fixed_schedule']:,.0f}\n"
            f"- Predictive: {cost['predictive']:,.0f}\n"
            f"- Predictive vs fixed: {cost['saving_vs_fixed_pct']:.1f}%\n"
            f"- Unplanned failures, fixed vs predictive: {cost['failures_fixed']} vs {cost['failures_predictive']}\n\n"
            f"**Signals driving predictions:** {', '.join(imp['feature'].head(3))}")


# This asks an online language model to rewrite the note, but only if a key is available. If anything fails, it returns nothing.
def llm_memo(facts):
    # Read the secret key from the computer's environment. No key means we skip this step.
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    try:
        # Send the facts and ask for a short manager's note that does not invent numbers.
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers={"Authorization": f"Bearer {key}"}, timeout=30,
                          json={"model": "llama-3.3-70b-versatile", "temperature": 0.2, "messages": [
                              {"role": "system", "content": "You write short maintenance memos for a plant manager. Use ONLY the numbers given. Never invent figures. State that costs are simulated assumptions."},
                              {"role": "user", "content": facts}]})
        # Pull the rewritten note out of the reply.
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        # If the call fails, fall back to the plain note instead of crashing.
        return None


# This writes the maintenance note to a file. It uses the rewritten version when that worked, otherwise the plain facts.
def write_memo(preds, cost, imp, path, synthetic=False):
    # Build the fact list from the predictions, costs, and important clues.
    facts = facts_text(preds, cost, imp)
    # Prefer the rewritten note. If that is missing, use a simple note that says the costs are a simulation.
    body = llm_memo(facts) or ("Maintenance recommendation (auto-generated from model output)\n\n" + facts +
                               "\n\nNote: cost figures come from a simulation using stated assumptions, not real plant data.")
    # If the numbers came from fake data, put a loud warning at the top.
    warn = "> WARNING: generated from SYNTHETIC data. Not a real result.\n\n" if synthetic else ""
    # Save the finished note.
    path.write_text(warn + body)
