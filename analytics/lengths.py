import numpy as np
import json
import re
import matplotlib.pyplot as plt


# with open("strings.json") as f:
#     strings = json.load(f)

with open("dev_gpt_prompts_v2.json") as f:
    strings_dg = json.load(f)

with open("strings_plus.json") as f:
    strings = json.load(f)

strings = set(list(map(lambda x: x.strip(), strings)))

prompt_lengths = {}
for prompt in strings:
    prompt = re.sub("\{[^\}]*\}", "", prompt)

    length = len(prompt)

    if length not in prompt_lengths:
        prompt_lengths[length] = 0
    prompt_lengths[length] += 1

prompt_lengths_dg = {}
for prompt in strings_dg:
    prompt = re.sub("\{[^\}]*\}", "", prompt)

    length = len(prompt)

    if length not in prompt_lengths_dg:
        prompt_lengths_dg[length] = 0
    prompt_lengths_dg[length] += 1

freq = {
    k: v
    for k, v in sorted(prompt_lengths.items(), key=lambda item: item[1], reverse=True)
}
freq_dg = {
    k: v
    for k, v in sorted(
        prompt_lengths_dg.items(), key=lambda item: item[1], reverse=True
    )
}

string_lengths = freq.keys()
string_lengths_dg = freq_dg.keys()
frequencies = freq.values()
frequencies_dg = freq_dg.values()

# Adjusting the number of bins manually to avoid the error with weighted data
num_bins = 45
# Calculating bin edges
data_combined = list(string_lengths) + list(string_lengths_dg)
min_bin = min(data_combined)
max_bin = max(data_combined)
bin_width = (max_bin - min_bin) / num_bins
bins = np.linspace(min_bin, max_bin, num_bins + 1)

# Define bar width as a fraction of bin width
bar_width_fraction = 0.35
bar_width = bin_width * bar_width_fraction

# Recreating the histogram with set number of bins and a logarithmic y-axis
fig, ax = plt.subplots(figsize=(10, 8))
# Plot the first histogram shifted to the left
ax.hist(
    string_lengths,
    weights=frequencies,
    bins=bins,
    color="skyblue",
    # edgecolor="black",
    width=bar_width,
    align="mid",
    label="Set (a-e)",
)

# Shift the bins to the right for the second histogram
bins_shifted = bins + bar_width

# Plot the second histogram
ax.hist(
    string_lengths_dg,
    weights=frequencies_dg,
    bins=bins_shifted,
    color="salmon",
    # edgecolor="black",
    width=bar_width,
    align="mid",
    label="Set (f)",
)

# Set y-axis to logarithmic scale
ax.set_yscale("log")

# Adding labels and title
ax.set_xlabel("Prompt Length", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
# ax.set_title("Frequency Distribution of String Lengths (Log Y Axis)", fontsize=16)

# Adding grid
ax.grid(True, axis="x", which="major", linestyle="--", linewidth=0.5)
ax.legend()

# Displaying the plot
plt.tight_layout()
plt.show()
