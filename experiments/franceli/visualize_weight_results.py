
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Issue #15 - Discourse Classifier Experiments
# Results from the standardized development dataset

models = [
    "0% Balanced",
    "25% Balanced",
    "50% Balanced",
    "100% Balanced"
]

accuracy = [0.5242, 0.5175, 0.4972, 0.3738]
macro_f1 = [0.3018, 0.3537, 0.3590, 0.3179]
weighted_f1 = [0.4804, 0.4890, 0.4849, 0.3991]

# Define bar positions
x = np.arange(len(models))
bar_width = 0.25

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Plot three metrics for each of the four models
bars_accuracy = ax.bar(
    x - bar_width,
    accuracy,
    bar_width,
    label="Accuracy",
    color="#4C78A8"
)

bars_macro = ax.bar(
    x,
    macro_f1,
    bar_width,
    label="Macro F1",
    color="#F58518"
)

bars_weighted = ax.bar(
    x + bar_width,
    weighted_f1,
    bar_width,
    label="Weighted F1",
    color="#54A24B"
)

# Add score labels above every bar
for bars in [bars_accuracy, bars_macro, bars_weighted]:
    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=3,
        fontsize=9
    )

# Highlight best Macro F1 model
best_index = int(np.argmax(macro_f1))

ax.annotate(
    "Best Macro F1",
    xy=(x[best_index], macro_f1[best_index]),
    xytext=(x[best_index] + 0.25, 0.62),
    arrowprops=dict(
        arrowstyle="->",
        color="black"
    ),
    fontsize=11,
    fontweight="bold"
)

# Graph formatting
ax.set_title(
    "Discourse Classification - Class Weight Comparison",
    fontsize=16,
    fontweight="bold",
    pad=20
)

ax.set_xlabel("Model Configuration", fontsize=12)
ax.set_ylabel("Evaluation Score", fontsize=12)

ax.set_xticks(x)
ax.set_xticklabels(models)

ax.set_ylim(0, 0.70)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

ax.set_axisbelow(True)
ax.legend(loc="upper right")

plt.tight_layout()

# Save visualization
output_dir = Path("experiments/franceli/results")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "class_weight_comparison.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

print("Visualization saved to:", output_file)

plt.show()
