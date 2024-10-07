import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc


if __name__ == "__main__":

    data_dir="data"
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, 'full_data_gpt35.pkl'), 'rb') as file:
        data_loaded = pickle.load(file)
    
    ranked_indices_gpt35 = data_loaded['ranked_indices']
    interest_binary_gpt35 = data_loaded['interest_binary']
    auc_values_gpt35 = data_loaded['auc_values']
    fpr_gpt35 = data_loaded['fpr']
    tpr_gpt35 = data_loaded['tpr']

    with open(os.path.join(data_dir, 'full_data_gpt4o.pkl'), 'rb') as file:
        data_loaded = pickle.load(file)
    
    ranked_indices_gpt4o = data_loaded['ranked_indices']
    interest_binary_gpt4o = data_loaded['interest_binary']
    auc_values_gpt4o = data_loaded['auc_values']
    fpr_gpt4o = data_loaded['fpr']
    tpr_gpt4o = data_loaded['tpr']


    with open(os.path.join(data_dir, 'full_data_ML.pkl'), 'rb') as file:
        data_loaded = pickle.load(file)
    
    topNprecision_avg_ML = data_loaded['topNprecision_avg']
    highInterestProb_ML = data_loaded['highInterestProb_ML']
    highInterestProb_rnd = data_loaded['highInterestProb_rnd']
    fpr_ML = data_loaded['fpr']
    tpr_ML = data_loaded['tpr']
    auc_value_ML = auc(fpr_ML, tpr_ML)


    topN_precision_gpt35 = [sum(interest_binary_gpt35[:i+1]) / (i+1) for i in range(len(interest_binary_gpt35))]
    max_precision_gpt35 = [max(interest_binary_gpt35[:i+1]) for i in range(len(interest_binary_gpt35))]

    topN_precision_gpt4o = [sum(interest_binary_gpt4o[:i+1]) / (i+1) for i in range(len(interest_binary_gpt4o))]
    max_precision_gpt4o = [max(interest_binary_gpt4o[:i+1]) for i in range(len(interest_binary_gpt4o))]

    # Parameters
    n_subsystem = 300  # Size of the subsystems
    total_size = len(interest_binary_gpt35)  # Assuming all have the same size
    iterations = 1000  # Number of iterations for averaging
    
    # Initialize arrays to hold the cumulative precision and max precision
    cumulative_topN_precision_gpt35 = np.zeros(n_subsystem)
    cumulative_max_precision_gpt35 = np.zeros(n_subsystem)
    
    cumulative_topN_precision_gpt4o = np.zeros(n_subsystem)
    cumulative_max_precision_gpt4o = np.zeros(n_subsystem)
    

    # Loop over the specified number of iterations
    for _ in range(iterations):
        # Select random indices for the subsystems without changing the order
        random_indices = np.sort(np.random.choice(total_size, n_subsystem, replace=False))
    
        # Extract subsystems while maintaining the order
        interest_binary_gpt35_sub = [interest_binary_gpt35[i] for i in random_indices]
        interest_binary_gpt4o_sub = [interest_binary_gpt4o[i] for i in random_indices]
       
        # Compute precision for the subsystems
        topN_precision_gpt35_sub = [sum(interest_binary_gpt35_sub[:i+1]) / (i+1) for i in range(len(interest_binary_gpt35_sub))]
        max_precision_gpt35_sub = [max(interest_binary_gpt35_sub[:i+1]) for i in range(len(interest_binary_gpt35_sub))]
    
        topN_precision_gpt4o_sub = [sum(interest_binary_gpt4o_sub[:i+1]) / (i+1) for i in range(len(interest_binary_gpt4o_sub))]
        max_precision_gpt4o_sub = [max(interest_binary_gpt4o_sub[:i+1]) for i in range(len(interest_binary_gpt4o_sub))]
    
        # Accumulate the results
        cumulative_topN_precision_gpt35 += np.array(topN_precision_gpt35_sub)
        cumulative_max_precision_gpt35 += np.array(max_precision_gpt35_sub)
    
        cumulative_topN_precision_gpt4o += np.array(topN_precision_gpt4o_sub)
        cumulative_max_precision_gpt4o += np.array(max_precision_gpt4o_sub)
    

    # Compute the averages
    average_topN_precision_gpt35 = cumulative_topN_precision_gpt35 / iterations
    average_max_precision_gpt35 = cumulative_max_precision_gpt35 / iterations
    
    average_topN_precision_gpt4o = cumulative_topN_precision_gpt4o / iterations
    average_max_precision_gpt4o = cumulative_max_precision_gpt4o / iterations

    overall_precision_gpt4o = sum(interest_binary_gpt4o) / len(interest_binary_gpt4o)
    
    # Create a vector with the same length as interest_binary_gpt4o, filled with the overall precision value
    topNprecision_avg_rnd = [overall_precision_gpt4o] * len(interest_binary_gpt4o)


    # Create a figure with three subplots
    fig = plt.figure(figsize=(18, 6))  # Adjusted for three subplots
    
    label_gpt35='GPT 3.5\n [text, 0-shot]'
    label_gpt4o='GPT 4o\n[text, 0-shot]'
    label_nn='Neural Net\n[graph, superv.]'
    label_rnd='random'
    
    # Subplot 1: ROC Curve
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(fpr_gpt35, tpr_gpt35, lw=3, label=f'{label_gpt35}\n(AUC={auc_values_gpt35[-1]:.3f})')
    ax1.plot(fpr_gpt4o, tpr_gpt4o, lw=3, label=f'{label_gpt4o}\n(AUC={auc_values_gpt4o[-1]:.3f})')
    ax1.plot(fpr_ML, tpr_ML, lw=3, label=f'{label_nn}\n(AUC={auc_value_ML:.3f})')
    ax1.plot([0, 1], [0, 1], color='grey', lw=3, linestyle='--',label=f'{label_rnd}\n(AUC={0.500:.3f})')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel('False Positive Rate', fontsize=14)  # Consistent font size
    ax1.set_ylabel('True Positive Rate', fontsize=14)  # Consistent font size
    ax1.set_title('Average ROC Curve', fontsize=16)  # Consistent title font size
    ax1.legend(loc="lower right", fontsize=12)  # Consistent legend font size
    ax1.grid(True)  # Add grid
    ax1.tick_params(axis='both', which='major', labelsize=14)  # Increase tick label size
    ax1.spines['top'].set_linewidth(1.5)  # Thicker frame line
    ax1.spines['right'].set_linewidth(1.5)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    fig.text(0.0, 0.96, '(a)', fontsize=18, weight='bold')

    # Subplot 2: Top-N Precision for First 300 N Values
    ax2 = fig.add_subplot(1, 3, 2)
    N = 300  # We will plot for the first 300 N values
    ax2.plot(range(1, N+1), average_topN_precision_gpt35[:N], lw=3, label=label_gpt35)
    ax2.plot(range(1, N+1), average_topN_precision_gpt4o[:N], lw=3, label=label_gpt4o)
    ax2.plot(range(1, len(topNprecision_avg_ML[:N])+1), topNprecision_avg_ML[:N], lw=3, label=label_nn)
    ax2.plot(range(1, len(topNprecision_avg_rnd[:N])+1), topNprecision_avg_rnd[:N], lw=3, linestyle='--', color='grey', label=label_rnd)
    ax2.set_xlim([1, N])
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('Sorted research suggestion', fontsize=14)  # Consistent font size
    ax2.set_ylabel('Precision', fontsize=14)  # Consistent font size
    ax2.set_title('Top-N Precision', fontsize=16)  # Consistent title font size
    ax2.legend(loc="upper right", fontsize=12)  # Consistent legend font size
    ax2.grid(True)  # Add grid
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.spines['top'].set_linewidth(1.5)
    ax2.spines['right'].set_linewidth(1.5)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    fig.text(0.33, 0.96, '(b)', fontsize=18, weight='bold')

    # Subplot 3: Top-N Precision for First 20 N Values (Max Precision)
    ax3 = fig.add_subplot(1, 3, 3)
    N = 10  # We will plot for the first 20 N values
    ax3.plot(range(1, N+1), average_max_precision_gpt35[:N], lw=3, label=label_gpt35)
    ax3.plot(range(1, N+1), average_max_precision_gpt4o[:N], lw=3, label=label_gpt4o)
    ax3.plot(range(1, len(highInterestProb_ML[:N])+1), highInterestProb_ML[:N], lw=3, label=label_nn)
    ax3.plot(range(1, len(highInterestProb_rnd[:N])+1), highInterestProb_rnd[:N], lw=3, linestyle='--', color='grey', label=label_rnd)
    ax3.set_xlim([1, N])
    ax3.set_ylim([0, 1])
    ax3.set_xlabel('Sorted research suggestion', fontsize=14)  # Consistent font size
    ax3.set_ylabel('Probability', fontsize=14)  # Consistent font size
    ax3.set_title('Top-N Success Probability', fontsize=16)  # Consistent title font size
    ax3.legend(loc="lower right", fontsize=12)  # Consistent legend font size
    ax3.grid(True)  # Add grid
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.spines['top'].set_linewidth(1.5)
    ax3.spines['right'].set_linewidth(1.5)
    ax3.spines['left'].set_linewidth(1.5)
    ax3.spines['bottom'].set_linewidth(1.5)
    fig.text(0.66, 0.96, '(c)', fontsize=18, weight='bold')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Directory and filename setup
    save_dir = 'figures'
    filename = 'Fig4.png'
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Full path to save the figure
    save_path = os.path.join(save_dir, filename)
    
    # Save the figure
    plt.savefig(save_path, dpi=300, format='png')

    # Show the plot
    plt.show()