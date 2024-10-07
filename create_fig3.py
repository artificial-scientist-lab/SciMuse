import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import pickle


def select_top_n_percent(all_data, N):
    # Calculate the number of entries to select
    num_entries = len(all_data['features'])
    top_n_count = int(num_entries * (N / 100))

    # Step 1: Sort the entries by 'impact' feature
    # Create a list of tuples where each tuple is (index, impact_value)
    impact_values = [(i, features[141]) for i, features in enumerate(all_data['features'])]

    # Sort this list by the impact_value in descending order
    sorted_by_impact = sorted(impact_values, key=lambda x: x[1], reverse=True)

    # Step 2: Select the top N% of these entries
    top_n_indices = [index for index, _ in sorted_by_impact[:top_n_count]]

    # Step 3: Create a new dictionary with these selected entries
    all_data_top_n = {
        'clarity': [all_data['clarity'][i] for i in top_n_indices],
        'interest': [all_data['interest'][i] for i in top_n_indices],
        'features': [all_data['features'][i] for i in top_n_indices]
    }

    return all_data_top_n

    
#    Node Features: 0-19 (20 elements)
#    Node Citation: 20-77 (58 elements)
#    Edge Features: 78-98 (21 elements)
#    Edge Citation: 99-140 (42 elements)
#    Subnet Ov.: 141-142 (2 elements)

if __name__ == "__main__":
    authortype = ['nat', 'soc']
    institutetype = ['nat', 'same', 'soc']
    suggestiontype = ['random', 'semnet']

    all_features = [0, 14, 20, 26, 75, 87, 137, 143]
    inset_range = [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
    all_titles = ['Degree of node A\n', 'PageRank of node A\n', 'Citation for node A\n', 'Total Citation for node A\n', 
                "Rank of 1-year citation increase\n for node B", 'Simpson similarity coefficient\nfor pair (A,B)', 
                'Total papers on concept A or B\n up to two years ago, minimum count', 
                "Semantic distance\n"]

    color_map = {100: 'blue', 50: 'green', 25: 'red'}  # Different colors for each percentage
    
    data_dir="data"
    file_path = os.path.join(data_dir, 'all_evaluation_data.pkl')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            all_data = pickle.load(file)
        print("'all_data' has been loaded from the pickle file.")
    else:
        print(f"{file_path} doesnt exist.")
        exit()

    # Set up the subplots
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(30, 15))  # Adjust the figure size as needed
    axes = axes.flatten()  # Flatten the array of axes to make indexing easier
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)'] 

    for i, (curr_feature, curr_range, curr_title) in enumerate(zip(all_features, inset_range, all_titles)):
        all_feature_vals_list = [ff[curr_feature] for ff in all_data['features']]
        all_feature_vals = np.array(all_feature_vals_list)
        mean_val = np.mean(all_feature_vals)
        std_val = np.std(all_feature_vals)

        for percentage in [25, 50, 100]:
            all_data_top_n = select_top_n_percent(all_data, percentage)
            interest_data = np.array(all_data_top_n['interest'])
            all_feature_vals_list = [ff[curr_feature] for ff in all_data_top_n['features']]
            all_feature_vals = np.array(all_feature_vals_list)

            # Normalize the values
            all_feature_vals_z = (all_feature_vals - mean_val) / std_val

            # Sort by all_feature_vals_z while keeping the correspondence with interest_data
            indices = np.argsort(all_feature_vals_z)
            sorted_feature_vals = all_feature_vals_z[indices]
            sorted_interest_data = interest_data[indices]

            num_parts = 50
            avg_interest_data, std_interest_data, avg_feature_vals = [], [], []
            for j in range(num_parts):
                index_start = j * len(sorted_feature_vals) // num_parts
                index_end = (j + 1) * len(sorted_feature_vals) // num_parts if j != num_parts - 1 else len(sorted_feature_vals)
                part_interest_data = sorted_interest_data[index_start:index_end]
                part_feature_vals = sorted_feature_vals[index_start:index_end]
                avg_interest_data.append(np.mean(part_interest_data))
                std_interest_data.append(np.std(part_interest_data, ddof=1) / np.sqrt(len(part_interest_data)))
                avg_feature_vals.append(np.mean(part_feature_vals))

            if percentage == 100:
                curr_label = 'All answers'
            else:
                curr_label = f'Top {percentage}% impact'
            axes[i].errorbar(avg_feature_vals, avg_interest_data, yerr=std_interest_data, fmt='o', capsize=5, color=color_map[percentage], label=curr_label, alpha=0.8)

        # Linear fit
        slope, intercept = np.polyfit(avg_feature_vals, avg_interest_data, 1)
        fit_line_linear = slope * np.array(avg_feature_vals) + intercept
        axes[i].plot(avg_feature_vals, fit_line_linear, 'grey', linestyle='--', linewidth=2, label='Linear Fit (all answers)')

        axes[i].set_title(curr_title, fontsize=24)
        axes[i].grid(True)
        if i==3:
            axes[i].legend(fontsize=22)
            
        axes[i].tick_params(axis='x', labelsize=23) 
        axes[i].set_ylim(1, 5)  # Adjust as necessary
        
        y_ticks = np.linspace(1, 5, 5)
        axes[i].set_yticks(y_ticks)
        axes[i].set_yticklabels(['{:.1f}'.format(y) for y in y_ticks], fontsize=23)
        axes[i].text(-0.11, 1.13, subplot_labels[i], transform=axes[i].transAxes, fontsize=28, fontweight='bold', va='top', ha='left')
        
        for spine in axes[i].spines.values():
            spine.set_linewidth(1.6)  # Adjust the thickness here

    # Set common labels more external to the plot area
    fig.text(0.5, 0.02, 'Normalized Feature Values', ha='center', va='center', fontsize=26)  # Common x-label, more below
    fig.text(0.02, 0.5, 'Average Interest', ha='center', va='center', rotation='vertical', fontsize=26)  # Common y-label, more left

    #plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])  # Adjust layout to not clip content
    fig_dir='figures'
    os.makedirs(fig_dir, exist_ok=True) 
    plt.tight_layout(rect=[0.03, 0.03, 1, 1]) 
    fig.subplots_adjust(hspace=0.25, wspace=0.16)        
    plt.savefig(os.path.join(fig_dir, 'Fig3.png'), format='png', dpi=300)
    plt.show()
    plt.close(fig)
    