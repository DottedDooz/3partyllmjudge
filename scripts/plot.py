import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

data_folder = 'path\\to\\data'

# 1. Load data
llm = pd.read_json(os.path.join(data_folder, 'llm_results.json'))
games = pd.read_csv(os.path.join(data_folder, 'tt_games_enriched.csv'))

# 2. Merge on game id & ai_witness_id
df = pd.merge(
    llm,
    games[['id', 'ai_model', 'ai_witness_id']],
    left_on=['game_id', 'ai_witness_id'],
    right_on=['id', 'ai_witness_id'],
    how='inner'
)

# 3. Compute AI pass (mistaken-for-human) rate
df['ai_success'] = df['llm_response'] != df['human_label']
df['human_success'] = df['llm_response'] == df['human_label']

# 3.5. Overall human win rate
human_rate = df['human_success'].mean()        # fraction
human_pct  = human_rate * 100                  # percent

# 4. Aggregate by model
agg = (
    df.groupby('ai_model')['ai_success']
      .agg(['mean', 'count'])
      .reset_index()
      .rename(columns={'mean':'pass_rate', 'count':'n_games'})
)
agg['pass_rate_pct'] = agg['pass_rate'] * 100
agg['error_pct'] = 1.96 * np.sqrt(
    agg['pass_rate'] * (1 - agg['pass_rate']) / agg['n_games']
) * 100

label_map = {
    "gpt-4o_minimal":       "GPT-4o\n(no persona)",
    "eliza":                "ELIZA",
    "gpt-4.5_minimal":      "GPT-4.5\n(no persona)",
    "llama-405b_minimal":   "LLaMa-3.1-405B\n(no persona)",
    "llama-405b_quinn":     "LLaMa-3.1-405B\n(with persona)",
    "gpt-4.5_quinn":        "GPT-4.5\n(with persona)",
}

# Create a new column for plotting
agg['display_name'] = agg['ai_model'].map(label_map)

# Drop any models we didn’t map (if any)
agg = agg.dropna(subset=['display_name'])

# 5. Keep only the six target models and enforce original order
display_order = [
    "GPT-4o\n(no persona)",
    "ELIZA",
    "GPT-4.5\n(no persona)",
    "LLaMa-3.1-405B\n(no persona)",
    "LLaMa-3.1-405B\n(with persona)",
    "GPT-4.5\n(with persona)"
]
reversed_order = list(reversed(display_order))

# Use pandas Categorical to enforce ordering
agg['display_name'] = pd.Categorical(
    agg['display_name'],
    categories=reversed_order,
    ordered=True
)
agg = agg.sort_values('display_name')


# 6. Prepare plotting data
models    = agg['display_name']
win_rates = agg['pass_rate_pct']
errors    = agg['error_pct']
n_games   = agg['n_games']
y_pos     = np.arange(len(models))
colors    = ['limegreen','gray','mediumorchid','lightskyblue','mediumblue','darkviolet']
total_n   = agg['n_games'].sum()

# 7. Plot
fig, ax = plt.subplots(figsize=(10, 6))
bar_height = 0.6

# Gray background strips
ax.barh(y_pos, [100]*len(models), height=bar_height, color='lightgray', zorder=0)

# Colored bars + error bars
ax.barh(
    y_pos, win_rates, xerr=errors, height=bar_height,
    color=colors,
    capsize=5, zorder=1
)

# Dynamic human-win line
ax.axvline(human_pct, color='gray', linestyle='--', linewidth=2, zorder=2)
ax.text(human_pct + 1, -0.5,
        f"Human Win Rate ({human_pct:.1f}%)",
        color='gray', fontsize=10, va='bottom')


# Labels & formatting
ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=11)
ax.set_xlabel("AI Win Rate", fontsize=12)
ax.set_title(
    "AI Model Pass Rates in Three-Party Turing Test (judged by 4o-mini)",
    fontsize=14
)
ax.set_xlim(0, 100)
for spine in ['top','right','left']:
    ax.spines[spine].set_visible(False)

# 8. Annotate actual n per model (multiline)
plt.text(99.5, -1.15, f"n = {total_n:,}", ha='right', va='bottom', fontsize=9)


plt.tight_layout()
plt.show()