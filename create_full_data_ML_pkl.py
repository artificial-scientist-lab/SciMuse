import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import time
import pickle
import torch
from torch import nn
import torch.nn.functional as F
import random
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from datetime import datetime

def print_log(log_string):
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    print_string=f"{formatted_time}: {log_string}"
    print(print_string)
    with open(log_file, 'a') as file:
        file.write(print_string+'\n')
    
    

class InterestPredictor(nn.Module):
    def __init__(self, input_features, neurons_per_layer, dropout_rate):
        super(InterestPredictor, self).__init__()
        
        num_layers = len(neurons_per_layer)
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_features, neurons_per_layer[0]))
        self.layers.append(nn.ReLU())
        # Optionally add dropout after the first layer or activation
        self.layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(1, num_layers):
            self.layers.append(nn.Linear(neurons_per_layer[i-1], neurons_per_layer[i]))
            self.layers.append(nn.ReLU())
            # Add dropout after each activation
            self.layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        # No dropout is applied to the output layer
        self.layers.append(nn.Linear(neurons_per_layer[-1], 1))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def compute_parameters(nn_structure, first_layer=1):
    # Add the input layer size at the beginning and the output size at the end
    layers = [first_layer] + nn_structure + [1]
    parameters = 0
    for i in range(1, len(layers)):
        parameters += (layers[i-1] * layers[i]) + layers[i]
    return parameters



def normalize_features(all_data):
    features_norm=[]

    for curr_feature in range(len(all_data['features'][0])):
        all_feature_vals_list = [ff[curr_feature] for ff in all_data['features']]
        all_feature_vals = np.array(all_feature_vals_list)


        # Normalize the values
        mean_val = np.mean(all_feature_vals)
        std_val = np.std(all_feature_vals)
        all_feature_vals_z = (all_feature_vals - mean_val) / std_val

        features_norm.append(all_feature_vals_z)
        
    features_norm=np.array(features_norm)
    return features_norm



def prepare_data(features_norm, interest_data, train_ratio, val_ratio, test_ratio):
    # Ensure the split ratios sum to 1
    
    # Shuffle the data
    indices = np.arange(len(interest_data))
    np.random.shuffle(indices)
    features_norm = features_norm[:, indices]
    interest_data = interest_data[indices]
    
    # Calculate split indices
    train_index = int(len(interest_data) * train_ratio)
    val_index = int(len(interest_data) * (train_ratio + val_ratio))
    
    # Split the data
    X_train = features_norm[:, :train_index].T
    y_train = interest_data[:train_index]
    
    X_val = features_norm[:, train_index:val_index].T
    y_val = interest_data[train_index:val_index]
    
    X_test = features_norm[:, val_index:].T
    y_test = interest_data[val_index:]
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    

    return X_train, y_train, X_val, y_val, X_test, y_test


def train(X_train, y_train, X_val, y_val, model, optimizer, epochs=100, patience=10, do_plot=False):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model = None
    
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        train_loss = criterion(outputs.squeeze(), y_train)
        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss.item())

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            predictions = model(X_val)
            val_loss = criterion(predictions.squeeze(), y_val)
            val_losses.append(val_loss.item())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict().copy()  # Ensure a deep copy is made for the model state
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        if epochs_no_improve == patience:
            #print(f'Early stopping triggered at epoch {epoch+1}')
            break

        #if (epoch+1) % 100 == 0:
        #    print(f'Epoch {epoch+1}, Training Loss: {train_loss.item()}, Val Loss: {val_loss.item()}')

    model.load_state_dict(best_model)

    return model



def evaluate_binary_classification(model, X_test, y_test, iteration=0, do_plot=False):
    #print(f'{torch.sum(y_test > 3).float()}/{len(y_test)}')
    model.eval()
    with torch.no_grad():
        raw_predictions = model(X_test)
        predictions_proba = torch.sigmoid(raw_predictions).squeeze()

    # Adjust labels if needed
    if sum((y_test > 3)) > 0:
        y_test_binary = (y_test > 3).int()



    # Get indices that would sort predictions_proba in descending order
    sorted_indices = torch.argsort(predictions_proba, descending=True)
    sorted_y_test_binary = y_test_binary[sorted_indices]
    
    random_order = torch.randperm(len(y_test_binary))
    random_y_test_binary = y_test_binary[random_order]
    

    # Calculate the cumulative sum of the sorted binary labels
    cumulative_sums = torch.cumsum(sorted_y_test_binary, dim=0)
    y_cummax=sorted_y_test_binary.cummax(dim=0)[0].numpy()
    rnd_cummax=random_y_test_binary.cummax(dim=0)[0].numpy()
    
    denominators = torch.arange(1, len(sorted_y_test_binary) + 1)
    
    # Calculate precision for each threshold
    precision = cumulative_sums.float() / denominators.float()
    precision_numpy = precision.numpy()

    # Calculate ROC AUC as well for comparison
    fpr, tpr, _ = roc_curve(y_test_binary.numpy(), predictions_proba.numpy())
    roc_auc = auc(fpr, tpr)

    if do_plot:
        # Plot Precision-Recall curve
        plt.figure(figsize=(12, 6))

        plt.plot(precision_numpy, color='blue', lw=2, label=f'Precision curve)')
        plt.xlabel('index')
        plt.ylabel('Precision')
        plt.title('Precision Curve')
        plt.legend(loc="lower left")
        plt.show()
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve by fpr, tpr')
        plt.legend(loc="lower right")
        plt.show()


    return roc_auc, precision_numpy, fpr, tpr, y_cummax, rnd_cummax




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


if __name__ == "__main__":
    # Configuration

    log_dir='logs_MLdata'
    os.makedirs(log_dir, exist_ok=True)        

    authortype = ['nat', 'soc']
    institutetype = ['nat', 'same', 'soc']
    suggestiontype = ['random', 'semnet']

    
    np.random.seed()
    random.seed()
    torch_seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(torch_seed)
    rrrseed=2
    np.random.seed(rrrseed)
    random.seed(rrrseed)
    torch.manual_seed(rrrseed)    
    
    CURR_ID=random.randint(10000000, 99999999)
    log_file=os.path.join(log_dir, f"logs_{CURR_ID}.txt")

    # Load and prepare your data outside the loop if it's the same for each iteration
    data_dir="data"
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, 'all_evaluation_data.pkl')

    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            all_data = pickle.load(file)
        print("'all_data' has been loaded from the pickle file.")
    else:
        print(f"{file_path} doesnt exist.")
        exit()


    percentage=100
    all_data_top_n = select_top_n_percent(all_data, percentage)
    all_data=all_data_top_n 

    interest_data = np.array(all_data['interest'])
    features_norm = normalize_features(all_data)
    
    n_features=25
    if True:
    #for n_features in list(range(10, 100, 5)):
        precision_all= np.array([])
        ycummax_all=np.array([])
        rnd_cummax_all=np.array([])
        
        fpr_all = np.array([])  # Initialize if not already
        tpr_all = np.array([])  # Initialize if not already        
        auc_scores = []
        std_of_the_mean=1

        curr_feature_list=[143, 6, 66, 76, 40, 24, 30, 15, 2, 72, 12, 36, 17, 10, 20, 34, 0, 4, 7, 14, 26, 16, 38, 5, 28, 68, 22, 8, 42, 13, 19, 9, 32, 70, 62, 64, 74, 18, 11, 3, 25, 1, 35, 135, 55, 23, 136, 27, 39, 53, 137, 31, 43, 60, 65, 112, 37, 29, 51, 123, 124, 117, 41, 125, 129, 130, 118, 119, 33, 50, 71, 131, 21, 63, 138, 54, 88, 52, 75, 132, 140, 59, 87, 56, 89, 103, 133, 77, 96, 139, 102, 73, 58, 69, 104, 67, 93, 48, 94, 90, 91, 97, 92, 122, 106, 99, 79, 100, 45, 61, 121, 95, 49, 142, 78, 80, 46, 98, 111, 120, 47, 101, 84, 44, 107, 128, 81, 134, 82, 85, 113, 116, 114, 127, 105, 115]
        #curr_feature_list=[8, 70, 66, 81, 39, 117, 125, 97, 40, 55, 17, 5, 46, 123, 48, 3, 41, 91, 92, 111, 114, 24, 135, 74, 95, 82, 10, 113, 87, 112, 124, 63, 4, 0, 27, 42, 143, 15]
        curr_feature_list=curr_feature_list[0:n_features]
        curr_neurons_per_layer_list=[50]
        curr_lr=0.003
        curr_train_ratio=0.75
        curr_dropout=0.2
        curr_weight_decay=0.0007

        hyperparameters=curr_neurons_per_layer_list, curr_feature_list, curr_lr, curr_train_ratio, curr_dropout, curr_weight_decay

        #hyperparameters=[[87], [8, 70, 66, 81, 39, 117, 125, 97, 40, 55, 17, 5, 46, 123, 48, 3, 41, 91, 92, 111, 114, 24, 135, 74, 95, 82, 10, 113, 87, 112, 124, 63, 4, 0, 27, 42, 143, 15], 0.00257, 0.7547, 0.21, 0.0007]
        #curr_neurons_per_layer_list, curr_feature_list, curr_lr, curr_train_ratio, curr_dropout, curr_weight_decay=hyperparameters


        curr_features_norm = features_norm[curr_feature_list, :]        
        print_log(f"hyperparameters={hyperparameters}\n")
        did_early_stop=False
        while len(auc_scores)<10 or std_of_the_mean>1/3*0.01:
            X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(curr_features_norm, interest_data, train_ratio=curr_train_ratio, val_ratio=0.9-curr_train_ratio, test_ratio=0.1)
    
            # Re-instantiate model and optimizer
            model = InterestPredictor(input_features=len(curr_features_norm), neurons_per_layer=curr_neurons_per_layer_list, dropout_rate=curr_dropout)
            optimizer = torch.optim.Adam(model.parameters(), lr=curr_lr, weight_decay=curr_weight_decay)
            
            criterion = nn.MSELoss()

            # Train the model
            #model = train(X_train, y_train, X_val, y_val, model, criterion, optimizer, epochs=1000, patience=200)
            model = train(X_train, y_train, X_val, y_val, model, optimizer, epochs=1000, patience=200)

            # Evaluate the model
            roc_auc, precision_numpy, fpr, tpr, y_cummax, rnd_cummax = evaluate_binary_classification(model, X_test, y_test, iteration=len(auc_scores), do_plot=False)
            
            fpr_all = np.concatenate((fpr_all, fpr))
            tpr_all = np.concatenate((tpr_all, tpr))
            
            if len(precision_all)==0:
                precision_all = precision_numpy
                ycummax_all =y_cummax
                rnd_cummax_all=rnd_cummax
            else:
                precision_all += precision_numpy
                ycummax_all +=y_cummax
                rnd_cummax_all+=rnd_cummax

            if roc_auc!=-1:
                auc_scores.append(roc_auc)
                std_of_the_mean=np.std(auc_scores)/np.sqrt(len(auc_scores))
                
                curr_val=(np.mean(auc_scores)+3*std_of_the_mean)
                        
                if len(auc_scores)%1==0:
                    #print_log('---')
                    print_log(f'{len(auc_scores)}: roc_auc={roc_auc:.4f} ({np.mean(auc_scores):.4f}+-{np.std(auc_scores)/np.sqrt(len(auc_scores)):.4f}))')


        # Sorting fpr_all and tpr_all by fpr
        indices_fpr = np.argsort(fpr_all)
        fpr_all_sorted = fpr_all[indices_fpr]
        tpr_all_sorted = tpr_all[indices_fpr]
        
        # Number of bins (N) and calculating the size of each bin
        N = 100  # Adjust N based on your requirements
        bin_size = len(indices_fpr) // N  # Using thresholds for binning precision
        
        # Initialize the bins for averaged data
        fpr_bin = np.zeros(N)
        tpr_bin = np.zeros(N)
        
        # Populate the bins by averaging the elements in each bin
        for i in range(N):
            start_index = i * bin_size
            end_index = start_index + bin_size
            fpr_bin[i] = np.mean(fpr_all_sorted[start_index:end_index])
            tpr_bin[i] = np.mean(tpr_all_sorted[start_index:end_index])
        
        
        precision_avg=precision_all/len(auc_scores)
        ycummax_avg=ycummax_all/len(auc_scores)
        rnd_cummax_avg=rnd_cummax_all/len(auc_scores)
        
        
        # Number of elements in precision_avg
        n_elements = len(precision_avg)
        
        # Calculate average precision using PyTorch
        avg_precision = precision_avg[-1]  # Calculate average precision
            
        
    fig = plt.figure(figsize=(18, 6))  # Adjusted for three subplots
    
    # ROC Curve
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(fpr_bin, tpr_bin, color='darkorange', lw=4, label=f'ML Selection (AUC={np.mean(auc_scores*100):.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=4, label='Random Selection')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel('False Positive Rate', fontsize=14)  # Increased font size
    ax1.set_ylabel('True Positive Rate', fontsize=14)  # Increased font size
    ax1.set_title('Average ROC Curve', fontsize=20)  # Increased title font size
    ax1.legend(loc="lower right", fontsize=12)  # Increased legend font size
    
    # Precision-Threshold Curve
    ax2 = fig.add_subplot(1, 3, 2)
    values = np.arange(1, len(precision_avg) + 1)
    ax2.plot(values, precision_avg, color='darkorange', lw=4, label='ML Selection')
    ax2.axhline(y=avg_precision.item(), color='navy', linestyle='--', lw=4, label='Random Selection')
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlim([1, len(precision_avg)])
    ax2.set_xlabel('Index of Research Suggestion', fontsize=14)  # Increased font size
    ax2.set_ylabel('Precision', fontsize=14)  # Increased font size
    ax2.set_title('Top-N Precision', fontsize=20)  # Increased title font size
    ax2.legend(loc="lower right", fontsize=12)  # Increased legend font size
    
    # Ycummax-Avg Curve
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(values[0:20], ycummax_avg[0:20], color='darkorange', lw=4, label='ML Selection')
    ax3.plot(values[0:20], rnd_cummax_avg[0:20], color='navy', linestyle='--', lw=4, label='Random Selection')
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlim([1, len(ycummax_avg[0:20])])
    ax3.set_xlabel('Number of Research Suggestion', fontsize=14)  # Increased font size
    ax3.set_ylabel('Probability', fontsize=14)  # Increased font size
    ax3.set_title('High-Interest Probability', fontsize=20)  # Increased title font size
    ax3.legend(loc="lower right", fontsize=12)  # Increased legend font size
    
    plt.tight_layout()
    #plt.savefig(f'best_model_{curr_neurons_per_layer_list[0]}_{percentage}_len_{len(curr_feature_list)}_LR_{curr_lr:.4f}_AUC_{np.mean(auc_scores):.4f}.png', dpi=300, format='png')
    
    plt.show()

    
    # Organize your data into a dictionary with the expected keys
    data_to_save = {
        'topNprecision_avg': precision_avg,        # Renamed for loading
        'highInterestProb_ML': ycummax_avg,        # Renamed for loading
        'highInterestProb_rnd': rnd_cummax_avg,    # Renamed for loading
        'fpr': fpr_bin,                            # Matches the expected key
        'tpr': tpr_bin                             # Matches the expected key
    }
    
    # Save the dictionary to a pickle file
 
    with open(os.path.join(data_dir,'full_data_ML.pkl'), 'wb') as file:
        pickle.dump(data_to_save, file)