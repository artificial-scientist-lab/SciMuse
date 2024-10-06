import os
import random
import pickle
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

def read_elo_results(result_dir):
    elo_results = []
    for file_name in os.listdir(result_dir):
        if file_name.startswith("ELO_results_") and file_name.endswith(".txt"):
            with open(os.path.join(result_dir, file_name), 'r') as file:
                lines = file.readlines()
                for line in lines:
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

def get_small_paper_string(papers):
    return "\n".join([f"{paper['title']} ({paper['year']})" for paper in papers])

def normalize_features(data):
    features = np.array(data['features'])
    return (features - np.mean(features, axis=0)) / np.std(features, axis=0)


def get_author_paper_titles(author_dir, author_name):
    # Construct the path to the author's papers.txt file
    author_papers_file = os.path.join(author_dir, 'author_paper_concept', author_name, 'papers.txt')
    
    # Initialize an empty list to store the paper titles
    paper_titles = []
    
    # Open and read the file with utf-8 encoding
    with open(author_papers_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        # Extract titles, which are the 1st, 4th, 7th, etc. lines
        paper_titles = [lines[i].strip() for i in range(0, len(lines), 3)]
    
    return paper_titles

def get_suggestion_text(suggestion_text_file):
    with open(suggestion_text_file, 'r', encoding='utf-8') as file:
        content = file.read()
        
        # Find the position of the "Project Title:" string
        start_index = content.find("Project Title:")
        
        # If the string is found, return the text from the start index
        if start_index != -1:
            return content[start_index:]
        else:
            return "Project Title: not found in the file."



def get_collaborator_name(author_new, suggestion_file_name):
    # Normalize both author_new and suggestion_file_name for case-insensitive comparison
    author_new_lower = author_new.lower()
    suggestion_file_name_lower = suggestion_file_name.lower()

    # Construct the expected prefix with the author name
    prefix = f"suggestion_{author_new_lower}_"
    
    # Find the position of the prefix in the file name
    prefix_position = suggestion_file_name_lower.find(prefix)
    
    if prefix_position != -1:
        # Extract the part after the prefix and before the '.txt' extension
        collaborator_name_with_extension = suggestion_file_name[prefix_position + len(prefix):]
        collaborator_name = collaborator_name_with_extension.rsplit('.txt', 1)[0]
        return collaborator_name
    
    # Return a message if the format is not as expected
    return "Collaborator name not found."

def get_survey_statistics_author(author_dir, author, suggestiontype, institutetype, authortype, do_get_suggestion_text=False):
    author_new = '_'.join([part.capitalize() for part in author.split('_')])
    
    author_inst_type_file=os.path.join(author_dir, 'chosen_researchers.txt')
    # Open the file and read the third line
    third_line=''
    with open(author_inst_type_file, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file, 1):
            if i == 3:
                third_line = line.strip()  # Use .strip() to remove newline characters
                break
    if 'NaturalScience' in third_line:
        author_institute_type='nat'
    if 'SocialScience' in third_line:
        author_institute_type='soc'
    if third_line=='':
        print('mistake!!!')
            
    norm_feature_file = os.path.join(author_dir, 'author_evaluation_result', f'new_features_{author}.csv')
    #norm_feature_file = os.path.join(author_dir, 'author_evaluation_result', f'norm_feature_{author_new}.csv')
    survey_result_file = os.path.join(author_dir, 'author_evaluation_result', f'survey_result_{author_new}.csv')
    
    
    suggestion_text_dir=os.path.join(author_dir, 'gpt4_response')
    
    
    if not os.path.exists(norm_feature_file) or not os.path.exists(survey_result_file):
        print(f"    Survey not completed for {author}: {norm_feature_file}")
        return

    with open(norm_feature_file, 'r', encoding='utf-8') as file1, open(survey_result_file, 'r', encoding='utf-8') as file2:
        norm_feature_reader = csv.reader(file1)
        survey_result_reader = csv.reader(file2)

        next(norm_feature_reader)  # Skip header
        next(survey_result_reader)

        author_clarity=[]
        author_interest=[]
        author_features=[]
        
        if do_get_suggestion_text==True:
            author_suggestion_text=[]
            author_own_papers=[]
            author_collaborator_papers=[]
            all_suggestion_text_file=[]
        
        
        for i in range(48):
            norm_feature_row = next(norm_feature_reader)
            survey_result_row = next(survey_result_reader)
            
            if do_get_suggestion_text==True:
                suggestion_file_name=survey_result_row[1]
                collaborator_name=get_collaborator_name(author_new, suggestion_file_name)

                suggestion_text_file=os.path.join(suggestion_text_dir, suggestion_file_name)
                curr_suggestion_text=get_suggestion_text(suggestion_text_file)
                


            question_type = norm_feature_row[0]
            collaborator_institute_type=survey_result_row[2]
            clarity=survey_result_row[8]
            interest = survey_result_row[9]
            if clarity!='':
                clarity = float(clarity)
            else:
                clarity=-1
            
            if interest!='':    
                interest = float(interest)
            else:
                interest=-1
            
            if (author_institute_type in authortype) and (question_type in suggestiontype) and (collaborator_institute_type in institutetype) and clarity!=-1 and interest!=-1:
                author_own_papers.append(get_author_paper_titles(author_dir, author_new))                
                author_collaborator_papers.append(get_author_paper_titles(author_dir, collaborator_name))                
                author_suggestion_text.append(curr_suggestion_text)
                all_suggestion_text_file.append(suggestion_text_file)
                
                author_clarity.append(clarity)
                author_interest.append(interest)
                curr_features=[]
                                
                for prop_index in range(6, 150):  # Columns 7 to 147
                    curr_features.append(float(norm_feature_row[prop_index]))                    
                author_features.append(curr_features)             
                


    if do_get_suggestion_text==True: 
        author_dict = {
            "clarity": author_clarity,
            "interest": author_interest,
            "features": author_features,
            "suggestion_text": author_suggestion_text,
            "own_paper_titles": author_own_papers,
            "collaborator_paper_titles": author_collaborator_papers,
            "suggestion_text_file": all_suggestion_text_file # for debugging
        }            

    else:
        author_dict = {
            "clarity": author_clarity,
            "interest": author_interest,
            "features": author_features
        }        
        
    return author_dict





def get_survey_statistics(suggestiontype, institutetype, authortype, do_get_suggestion_text=False):
    # Path to the human_evaluation directory and figure storage directory
    human_evaluation_dir = 'human_evaluation'
    
    #print(f'feature_idx_list: {feature_idx_list}')

    if do_get_suggestion_text==True:
        total_data = {
            "clarity": [],
            "interest": [],
            "features": [],
            "suggestion_text": [],
            "own_paper_titles": [],
            "collaborator_paper_titles": [],
            "suggestion_text_file": []
        }
    else:
        total_data = {
            "clarity": [],
            "interest": [],
            "features": []
        }
    authors = [name for name in os.listdir(human_evaluation_dir) if os.path.isdir(os.path.join(human_evaluation_dir, name))]
    
    #print(authors)
    
    for author in authors:
        author_dir = os.path.join(human_evaluation_dir, author)
        author_data=get_survey_statistics_author(author_dir, author, suggestiontype, institutetype, authortype, do_get_suggestion_text=do_get_suggestion_text)
        #print(f'{author}: {author_data}')
        if author_data is not None:
            for key in author_data:
                total_data[key].extend(author_data[key])
    
    return total_data


if __name__ == "__main__":
    # Configuration
    for use_gpt4o in [True, False]:
    
        log_dir = 'logs_gpt'
        os.makedirs(log_dir, exist_ok=True)

        authortype = ['nat', 'soc']
        institutetype = ['nat', 'same', 'soc']
        suggestiontype = ['random', 'semnet']
    
        rrrseed = 2
        np.random.seed(rrrseed)
        random.seed(rrrseed)
        
        CURR_ID = random.randint(10000000, 99999999)
        log_file = os.path.join(log_dir, f"logs_{CURR_ID}.txt")
    
        # Load and prepare your data outside the loop if it's the same for each iteration
        file_path = 'all_evaluation_data.pkl'
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                all_data = pickle.load(file)
            print("'all_data' has been loaded from the pickle file.")
        else:
            all_data = get_survey_statistics(suggestiontype=suggestiontype, institutetype=institutetype, authortype=authortype, do_get_suggestion_text=True)
            with open(file_path, 'wb') as file:
                pickle.dump(all_data, file)
    
        # Main body
        num_of_samples = len(all_data['interest'])
        smaller_data = all_data
        
        pkl_dir = 'gpt_ml_data'
        elo_dir= "elo_data"
        os.makedirs(pkl_dir, exist_ok=True)
        os.makedirs(elo_dir, exist_ok=True)
        if use_gpt4o:
            #result_dir = 'results_4o'
            file_name=os.path.join(pkl_dir,'full_data_gpt4o.pkl')
            elo_file=os.path.join(elo_dir,'elo_data_gpt4o.pkl')
        else:
            #result_dir = 'results_gpt35'
            file_name=os.path.join(pkl_dir,'full_data_gpt35.pkl')
            elo_file=os.path.join(elo_dir,'elo_data_gpt35.pkl')
            
    
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
            