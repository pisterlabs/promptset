from matplotlib import pyplot as plt
import numpy as np

# Provided data
data1 = {
    "en": 19457,
    "zh": 386,
    "fr": 298,
    "de": 261,
    "ja": 186,
    "es": 181,
    "ru": 180,
    "pt": 135,
    "it": 80,
    "ko": 80,
    "kn": 74,
    "pl": 65,
    "nl": 47,
    "sv": 42,
    "ta": 37,
    "ca": 32,
    "tr": 22,
    "da": 22,
    "fa": 20,
    "id": 19,
    "ar": 19,
    "uk": 17,
    "si": 16,
    "mk": 14,
    "no": 13,
    "bn": 12,
    "th": 11,
    "hi": 11,
    "sr": 11,
    "ce": 11,
    "eo": 10,
}

data2 = {
    "en": 11562,
    "ja": 659,
    "zh": 368,
    "ko": 176,
    "pt": 150,
    "de": 119,
    "da": 114,
    "ru": 66,
    "es": 55,
    "fr": 45,
    "ce": 44,
    "ro": 41,
    "sv": 21,
    "it": 17,
    "ca": 16,
    "nl": 15,
    "no": 10,
    "hu": 10,
}

# Initialize the combined structure with zeros
combined_data = {code: (0, 0) for code in set(data1) | set(data2)}

# Fill in the counts from the provided data
for code in combined_data:
    combined_data[code] = (data1.get(code, 0), data2.get(code, 0))

# Sort combined_data by total count in descending order
combined_data = {
    code: counts
    for code, counts in sorted(
        combined_data.items(), key=lambda item: item[1][0] + item[1][1], reverse=True
    )
}

# Unpack the languages and counts into separate lists
languages, count_pairs = zip(*combined_data.items())
counts1, counts2 = zip(*count_pairs)

# Plotting the histograms
fig, ax = plt.subplots(figsize=(10, 8))

# Define the width of the bars
bar_width = 0.35

# Calculate the bar positions
index = np.arange(len(languages))
bar1_positions = index - bar_width / 2
bar2_positions = index + bar_width / 2

# Plot the bars
bar1 = ax.bar(bar1_positions, counts1, bar_width, label="Set (a-e)", color="skyblue")
bar2 = ax.bar(bar2_positions, counts2, bar_width, label="Set (f)", color="salmon")

# Add labels, title, and legend
ax.set_xlabel("Language Code")
ax.set_ylabel("Count")
ax.set_title("Comparative Histogram of Language Code Counts")
ax.set_xticks(index)
ax.set_xticklabels(languages, rotation=90)
ax.legend()
ax.set_yscale("log")

# Final layout adjustments

plt.tight_layout()

# Show the plot
plt.show()
