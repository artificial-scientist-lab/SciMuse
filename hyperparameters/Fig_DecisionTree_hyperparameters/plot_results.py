import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

script_path = os.path.abspath(os.path.dirname(__file__))
# Change current working directory to the script's location
os.chdir(script_path)
print("current path:", script_path)
# List of num_features values
num_features_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]

# Set up the figure and axes for a 3x3 grid
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))

# Set the color normalization range
vmin = 0.55
vmax = 0.62

# Flatten the axes array for easy iteration
axes = axes.flatten()

for idx, num_features in enumerate(num_features_list):
    print(num_features)
    data = []
    # Read and parse the file
    try:
        with open(f'all_results_{num_features}.txt', 'r', encoding='latin-1') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                # Regex pattern to extract n, m, and Mean AUC
                pattern = r'\((\d+),\s*(\d+)\):\s*Test AUC=[\d.]+\s*\(Mean AUC:\s*([\d.]+)\s*±\s*[\d.]+\)'
                match = re.match(pattern, line)
                if match:
                    n = int(match.group(1))
                    m = int(match.group(2))
                    mean_auc = float(match.group(3))
                    data.append({'max_depth': n, 'min_samples_leaf': m, 'Mean AUC': mean_auc})
                else:
                    print(f"Line not matched in file {num_features}: {line}")
    except FileNotFoundError:
        print(f"File all_results_{num_features}.txt not found.")
        continue  # Skip to the next num_features value

    # Create a DataFrame
    df = pd.DataFrame(data)
    
    if df.empty:
        print(f"No data found in file all_results_{num_features}.txt.")
        continue

    # Pivot the DataFrame to create a table
    table = df.pivot(index='max_depth', columns='min_samples_leaf', values='Mean AUC')

    # Create the heatmap with reversed colormap and customized annotations
    sns.heatmap(
        table,
        ax=axes[idx],
        annot=True,
        fmt=".3f",
        cmap='YlGnBu_r',  # Reversed colormap
        vmin=vmin,
        vmax=vmax,
        annot_kws={"size": 10, "weight": "bold"},  # Bold and larger font
        cbar=False  # We'll add a single colorbar later
    )

    # Set plot labels and title
    axes[idx].set_title(f'{num_features} features')
    axes[idx].set_xlabel('min_samples_leaf (m)')
    axes[idx].set_ylabel('max_depth (n)')


# Add a single colorbar for all heatmaps
fig.subplots_adjust(right=0.9)  # Adjust the right boundary of the figure
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # x, y, width, height
norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap='YlGnBu_r', norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax)

# Adjust layout and set the overall title
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the colorbar
plt.suptitle('Mean AUC for Different Hyper-parameters of Decision Tree', fontsize=16, y=1.02)

# Save the figure as a high-resolution PNG file
plt.savefig('mean_auc_heatmaps_highres.png', dpi=300, bbox_inches='tight')

# If you want to display the plot as well, you can uncomment the following line:
plt.show()