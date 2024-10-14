import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# Get the directory where this script is located
script_path = os.path.abspath(os.path.dirname(__file__))

# Change current working directory to the script's location
os.chdir(script_path)
print("current path:", script_path)

# List of num_features values
num_features_list = [5, 15, 25, 35, 45]

# Set up the figure
fig = plt.figure(figsize=(25, 10))

# Set the color normalization range based on your data
vmin = 0.61
vmax = 0.65

# First row: Heatmaps for num_features
axes = []
for idx, num_features in enumerate(num_features_list):
    # Calculate position for each subplot
    left = 0.05 + idx * 0.18
    bottom = 0.55
    width = 0.16
    height = 0.4
    ax = fig.add_axes([left, bottom, width, height])
    axes.append(ax)
    print(f"Processing num_features={num_features}")
    data = []
    # Read and parse the file
    try:
        with open(f'all_results_{num_features}_0.003.txt', 'r', encoding='latin-1') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                # Regex pattern to extract n, m, and Mean AUC
                pattern = r'\((\d+),\s*(\d+)\):\s*Test AUC=[\d.]+\s*\(Mean AUC:\s*([\d.]+)\s*±\s*[\d.]+\)'
                match = re.match(pattern, line)
                if match:
                    n = int(match.group(1))  # Number of layers
                    m = int(match.group(2))  # Number of neurons per layer
                    mean_auc = float(match.group(3))
                    data.append({'layers': n, 'neurons': m, 'Mean AUC': mean_auc})
                else:
                    print(f"Line not matched in file {num_features}: {line}")
    except FileNotFoundError:
        print(f"File all_results_{num_features}_0.003.txt not found.")
        continue  # Skip to the next num_features value

    # Create a DataFrame
    df = pd.DataFrame(data)

    if df.empty:
        print(f"No data found in file all_results_{num_features}_0.003.txt.")
        continue

    # Pivot the DataFrame to create a table
    table = df.pivot(index='layers', columns='neurons', values='Mean AUC')

    # Check if the table is empty
    if table.empty:
        print(f"No data to plot for num_features={num_features}.")
        continue

    # Create the heatmap with reversed colormap and customized annotations
    sns.heatmap(
        table,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap='YlGnBu_r',
        vmin=vmin,
        vmax=vmax,
        annot_kws={"size": 12, "weight": "bold"},
        cbar=False
    )

    # Set plot labels and title
    ax.set_title(f'{num_features} features', fontsize=14, y=1.02)
    ax.set_xlabel('Neurons per Layer (m)', fontsize=12)
    ax.set_ylabel('Number of Layers (n)', fontsize=12)

# Second row: Heatmaps for Learning Rate and Dropout analysis at num_features=25

# Learning Rate Heatmap
ax_lr = fig.add_axes([0.22, 0.25, 0.16, 0.15])

# Data for LR analysis
mean_aucs_lr = [0.648, 0.648, 0.648]
learning_rates = [0.001, 0.003, 0.009]
columns_lr = [str(lr) for lr in learning_rates]
data_lr = [mean_aucs_lr]

# Create a DataFrame for LR analysis
df_lr = pd.DataFrame(data_lr, index=['AUC'], columns=columns_lr)

# Create the heatmap for Learning Rate
sns.heatmap(
    df_lr,
    ax=ax_lr,
    annot=True,
    fmt=".3f",
    cmap='YlGnBu_r',
    vmin=vmin,
    vmax=vmax,
    cbar=False,
    annot_kws={"size": 14, "weight": "bold"}
)

# Set labels and title for LR heatmap
ax_lr.set_title('Mean AUC for Learning Rates\n(25 features, 1 layer, 50 neurons)', fontsize=12, y=1.1)
ax_lr.set_xlabel('Learning Rate', fontsize=10)
ax_lr.set_ylabel('')
ax_lr.set_yticks([])

# Dropout Rate Heatmap
ax_dr = fig.add_axes([0.62, 0.25, 0.16, 0.15])

# Data for Dropout analysis
mean_aucs_dr = [0.647, 0.648, 0.649]
dropout_rates = [0.1, 0.2, 0.3]
columns_dr = [str(dr) for dr in dropout_rates]
data_dr = [mean_aucs_dr]

# Create a DataFrame for Dropout analysis
df_dr = pd.DataFrame(data_dr, index=['AUC'], columns=columns_dr)

# Create the heatmap for Dropout Rate
sns.heatmap(
    df_dr,
    ax=ax_dr,
    annot=True,
    fmt=".3f",
    cmap='YlGnBu_r',
    vmin=vmin,
    vmax=vmax,
    cbar=False,
    annot_kws={"size": 14, "weight": "bold"}
)

# Set labels and title for Dropout heatmap
ax_dr.set_title('Mean AUC for Dropout Rates\n(25 features, 1 layer, 50 neurons)', fontsize=12, y=1.1)
ax_dr.set_xlabel('Dropout Rate', fontsize=10)
ax_dr.set_ylabel('')
ax_dr.set_yticks([])

# Adjust the colorbar to cover both rows and move it further to the right
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.75])

norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap='YlGnBu_r', norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label='Mean AUC')

# Set overall title higher to avoid overlap
plt.suptitle('Mean AUC for Different Hyper-parameters of Neural Network', fontsize=18, y=1.05)

# Save the figure as a high-resolution PNG file
plt.savefig('mean_auc_heatmaps_nn_with_lr_dropout_single_colorbar.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
