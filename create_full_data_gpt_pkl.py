import os
import random
import pickle
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


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


def normalize_features(data):
    features = np.array(data['features'])
    return (features - np.mean(features, axis=0)) / np.std(features, axis=0)



if __name__ == "__main__":
    # Configuration
    for use_gpt4o in [True, False]:
    
        authortype = ['nat', 'soc']
        institutetype = ['nat', 'same', 'soc']
        suggestiontype = ['random', 'semnet']
    
        rrrseed = 2
        np.random.seed(rrrseed)
        random.seed(rrrseed)
    
        data_dir="data"
        os.makedirs(data_dir, exist_ok=True)
        # Load and prepare your data outside the loop if it's the same for each iteration
        file_path = os.path.join(data_dir, 'all_evaluation_data.pkl')
        
        with open(file_path, 'rb') as file:
            all_data = pickle.load(file)
        print("'all_data' has been loaded from the pickle file.")

        # Main body
        num_of_samples = len(all_data['interest'])
        smaller_data = all_data
        
        if use_gpt4o:
            #result_dir = 'results_4o'
            file_name=os.path.join(data_dir,'full_data_gpt4o.pkl')
            elo_file=os.path.join(data_dir,'elo_data_gpt4o.pkl')
        else:
            #result_dir = 'results_gpt35'
            file_name=os.path.join(data_dir,'full_data_gpt35.pkl')
            elo_file=os.path.join(data_dir,'elo_data_gpt35.pkl')
            
    
        # Initialize ELO scores and match counts if file doesn't exist
        elo_scores = [1400] * num_of_samples
        match_counts = [0] * num_of_samples
        
        with open(elo_file, 'rb') as file:
            elo_results = pickle.load(file)
                
        
        # Update ELO scores based on results
        for id1, id2, winner in elo_results:
            elo_scores = update_elo(elo_scores, id1, id2, winner)
            match_counts[id1] += 1
            match_counts[id2] += 1
    
        interest_data = np.array(smaller_data['interest'])
        features_norm = normalize_features(smaller_data)
        
        # Ranking suggestions by ELO from large to small
        ranked_indices = np.argsort(elo_scores)[::-1]
        
        # High interest is defined as 4 or 5, low interest as 1, 2, or 3
        interest_binary = [1 if interest_data[i] >= 4 else 0 for i in ranked_indices]
        
        # Compute AUC
        auc = roc_auc_score(interest_binary, np.sort(elo_scores)[::-1])
    
        print(f"AUC: {auc}")
        
        # Save the results
        total_matches = sum(match_counts) // 2

        # Plot the ROC curve
        fpr, tpr, _ = roc_curve(interest_binary, np.sort(elo_scores)[::-1])
        
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        plt.show()
        plt.close()
    
            
        # Create a dictionary with the required keys
        data_to_save = {
            'ranked_indices': ranked_indices,        # Your ranked indices data
            'interest_binary': interest_binary,      # Your binary interest data
            'auc_values': [auc],                # Your AUC values
            'fpr': fpr,                              # Your false positive rate data
            'tpr': tpr                               # Your true positive rate data
        }
        
        # Save the dictionary to a pickle file
        with open(file_name, 'wb') as file:
            pickle.dump(data_to_save, file)
            