import numpy as np
import json

# import langdetect
from ftlangdetect import detect
import re
import matplotlib.pyplot as plt

# with open("strings.json") as f:
#     strings = json.load(f)

with open("dev_gpt_prompts_v2.json") as f:
    strings_dg = json.load(f)

with open("strings_plus.json") as f:
    strings = json.load(f)

freq_dg = {}
for prompt in strings_dg:
    prompt_l = re.sub("\{[^\}]*\}", "", prompt)

    if len(prompt) < 6:
        continue

    try:
        lang = detect(prompt_l.split("\n")[0])["lang"]
    except Exception as e:
        lang = "error"

    freq_dg[lang] = freq_dg.get(lang, 0) + 1


freq_dg = {
    k: v for k, v in sorted(freq_dg.items(), key=lambda item: item[1], reverse=True)
}
freq_dg = {k[:2]: v for k, v in freq_dg.items() if v >= 10}

freq = {}
for prompt in strings:
    prompt_l = re.sub("\{[^\}]*\}", "", prompt)

    if len(prompt) < 6:
        continue

    try:
        lang = detect(prompt_l.split("\n")[0])["lang"]
    except Exception as e:
        lang = "error"

    freq[lang] = freq.get(lang, 0) + 1


freq = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True)}
freq = {k[:2]: v for k, v in freq.items() if v >= 10}

print(freq)
print(freq_dg)

string_lengths = freq.keys()
frequencies = freq.values()
string_lengths_dg = freq_dg.keys()
frequencies_dg = freq_dg.values()

# Adjusting the number of bins manually to avoid the error with weighted data
num_bins = len(freq)
data_combined = list(string_lengths) + list(string_lengths_dg)
min_bin = 0
max_bin = len(data_combined)
bin_width = (max_bin - min_bin) / num_bins
bins = np.linspace(min_bin, max_bin, num_bins + 1)

# Define bar width as a fraction of bin width
bar_width_fraction = 0.5
bar_width = bin_width * bar_width_fraction

# Recreating the histogram with set number of bins and a logarithmic y-axis
fig, ax = plt.subplots(figsize=(10, 8))
# Plot the first histogram shifted to the left
ax.hist(
    string_lengths,
    weights=frequencies,
    bins=bins,
    color="skyblue",
    edgecolor="black",
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
    edgecolor="black",
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
ax.grid(True, axis="y", which="major", linestyle="--", linewidth=0.5)
ax.legend()

# Displaying the plot
plt.tight_layout()
plt.show()
