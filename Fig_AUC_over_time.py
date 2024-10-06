import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def read_elo_results(file_path):
    elo_results = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if not len(line)==1:
                id1, id2, winner = line.strip().split(',')
                elo_results.append((int(id1), int(id2), int(winner)))
    return elo_results

def update_elo(elo_scores, id1, id2, winner, K=32):
    # Compute expected scores
    R1 = 10**(elo_scores[id1] / 400)
    R2 = 10**(elo_scores[id2] / 400)
    E1 = R1 / (R1 + R2)
    E2 = R2 / (R1 + R2)
    
    # Update scores
    if winner == 1:
        S1, S2 = 1, 0
    else:
        S1, S2 = 0, 1
    
    elo_scores[id1] = elo_scores[id1] + K * (S1 - E1)
    elo_scores[id2] = elo_scores[id2] + K * (S2 - E2)
    
    return elo_scores

if __name__ == "__main__":
    
    file_path = 'all_evaluation_data.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            all_data = pickle.load(file)
        print("'all_data' has been loaded from the pickle file.")
    else:
        print(f"{file_path} doesnt exist.")
        exit()
        
    num_of_samples = len(all_data['interest'])
    all_auc_labels=['GPT4o mini', 'GPT4o', 'GPT 3.5']
    result_files = ['combined_ELO_results_4omini.txt', 'combined_ELO_results_4o.txt', 'combined_ELO_results_35.txt']
    
    auc_elo_dir = 'auc_elo_data'
    os.makedirs(auc_elo_dir, exist_ok=True)
    all_auc_evolutions=[]
    for result_file in result_files:
        elo_scores = [1400] * num_of_samples
        match_counts = [0] * num_of_samples        
        elo_results = read_elo_results(os.path.join(auc_elo_dir, result_file))
        #random.shuffle(elo_results)
    
        # Prepare interest data and other relevant variables
        interest_data = np.array(all_data['interest'])
        
        # Initialize list to store AUC values
        auc_values = []
        
        # Update ELO scores based on results and compute AUC after each update
        for idx, (id1, id2, winner) in enumerate(elo_results):
            elo_scores = update_elo(elo_scores, id1, id2, winner)
            match_counts[id1] += 1
            match_counts[id2] += 1
        
            # Compute AUC after every 10th iteration
            if (idx + 1) % 1 == 0 or idx == len(elo_results) - 1:
                # Compute AUC
                ranked_indices = np.argsort(elo_scores)[::-1]
                interest_binary = [1 if interest_data[i] >= 4 else 0 for i in ranked_indices]
                auc = roc_auc_score(interest_binary, np.sort(elo_scores)[::-1])
                auc_values.append(auc)
                print(f'{idx + 1}/{len(elo_results)}: {auc}')
    
        all_auc_evolutions.append(auc_values)
        
    # Plot AUC values over the course of the tournament
    plt.figure()
    plt.plot(all_auc_evolutions[0], label=f'AUC over time ({all_auc_labels[0]})')
    plt.plot(all_auc_evolutions[1], label=f'AUC over time ({all_auc_labels[1]})')
    plt.plot(all_auc_evolutions[2], label=f'AUC over time ({all_auc_labels[2]})')
    plt.xlabel('Match Number')
    plt.ylabel('AUC')
    plt.title('AUC over the course of the ELO tournament')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    save_dir = 'figures'
    os.makedirs(save_dir, exist_ok=True)

    auc_plot_file = os.path.join(save_dir, "auc_over_time_final.png")
    plt.savefig(auc_plot_file, dpi=300, format='png')
    plt.show()
    plt.close()

    print(f"AUC over time plot saved to {auc_plot_file}")
