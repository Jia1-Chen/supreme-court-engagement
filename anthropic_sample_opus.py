import os
import pandas as pd
from anthropic import Anthropic

# --- 1. Initialize client ---
client = Anthropic(api_key='PUT YOUR API HERE')

# --- 2. Load your dataset ---
df = pd.read_csv("samples/30_pairs_dissent_1.csv")

# --- 3. Number of repetitions ---
num_runs = 5

# --- 4. Outer loop for time index ---
for run_idx in range(num_runs):
    output_dir = f"samples/anthropic_opus/responses_{run_idx}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=== Starting run {run_idx}, saving to {output_dir} ===\n")
    
    # --- 5. Inner loop over each case ---
    for i, row in df.iterrows():  
        row = df.loc[i]
        majority_text = row['majority_text']
        dissent_text = row['dissent_text']
        official_citation = row['official citation']
        
        prompt = (
            "You will be provided with a U.S. Supreme Court opinion, which consists of a majority opinion and a dissenting opinion. "
            "Your task is to evaluate the extent to which the dissent is 'talking with' the majority opinion or 'talking past' it.\n\n"
            "Definitions:\n"
            "Talking with (High Score: 5): The dissent directly engages with the majority's reasoning, addresses specific legal arguments, "
            "and attempts to rebut the key points in a way that demonstrates meaningful dialogue.\n"
            "Talking past (Low Score: 1): The dissent focuses on different issues, ignores key majority reasoning, "
            "or relies on a separate legal framework with little direct engagement.\n\n"
            "Scoring Criteria (1–5 Scale):\n"
            "5 – Strong engagement: The dissent thoroughly addresses the majority's reasoning, cites the same precedents and statutory interpretations, "
            "and provides a detailed rebuttal.\n"
            "4 – Substantial engagement: The dissent engages significantly with the majority's reasoning, responding to key points, but may also introduce broader concerns.\n"
            "3 – Moderate engagement: The dissent addresses some of the majority's arguments but also shifts focus to independent issues or alternative perspectives.\n"
            "2 – Minimal engagement: The dissent briefly acknowledges the majority's reasoning but mostly introduces independent legal arguments.\n"
            "1 – No engagement: The dissent is largely unrelated to the majority opinion, relying on a completely separate framework or ignoring key points.\n\n"
            f"Majority:\n{majority_text}\n\nDissent:\n{dissent_text}\n\n"
            "Return only a JSON object with two keys: 'score' (integer from 1 to 5) and 'reasoning' (a short explanation)."
        )
        
        try:
            response = client.messages.create(
                model="claude-opus-4-1-20250805",
                max_tokens=4096,
                system="You are an expert legal analyst evaluating Supreme Court opinions.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            result_text = response.content[0].text.strip()
            
            # --- Save each response ---
            filename = os.path.join(output_dir, f"response_{i}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Official Citation: {row['official citation']}\n\n")
                f.write(result_text)
            
            print(f"Saved response to {filename}")
            
        except Exception as e:
            print(f"Error processing row {i} in run {run_idx}: {e}")