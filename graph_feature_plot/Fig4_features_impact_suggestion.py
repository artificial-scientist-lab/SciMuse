import os
import csv
import numpy as np
import matplotlib.pyplot as plt

DATASET_FILE = 'evaluation_feature_dataset.csv'


def load_features(dataset_file, suggestiontype, institutetype, authortype):
    all_features = []

    with open(dataset_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # skip header

        for row in reader:
            author_institute_type = row[0]
            question_type = row[1]
            collaborator_institute_type = row[2]
            clarity = float(row[3])
            interest = float(row[4])

            if (author_institute_type in authortype and question_type in suggestiontype
                    and collaborator_institute_type in institutetype
                    and clarity != -1 and interest != -1):
                all_features.append([float(v) for v in row[5:]])

    return all_features


def main():
    os.makedirs('Figures', exist_ok=True)

    authortype = ['nat', 'soc']
    institutetype = ['nat', 'same', 'soc']
    suggestiontype = ['random', 'semnet']

    # These are the same x-axis features used in Fig4_features.py.
    all_features = [0, 14, 20, 26, 75, 87, 137, 143]

    all_titles = [
        'Degree of node A',
        'PageRank of node A',
        'Citation for node A',
        'Total Citation for node A',
        'Rank of 1-year citation increase\nfor node B',
        'Simpson similarity coefficient\nfor pair (A,B)',
        'Total papers on concept A or B\nuntil two years ago, minimum count',
        'Semantic distance'
    ]

    # The x-axis ranges used by the original average-interest figure.
    x_axis_ranges = [
        (-1.2192, 3.3697),
        (-1.0807, 4.8638),
        (-0.8310, 4.7729),
        (-0.8127, 6.1982),
        (-4.0856, 0.9498),
        (-2.1933, 1.8363),
        (-0.8736, 7.8288),
        (-2.3493, 1.9249)
    ]

    all_data = load_features(
        DATASET_FILE,
        suggestiontype=suggestiontype,
        institutetype=institutetype,
        authortype=authortype
    )

    # f142 is features[141]. It contains output(A,B) + output(B,A), so divide by 2.
    average_nn_output = np.array([features[141] / 2 for features in all_data])

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(28, 14))
    axes = axes.flatten()

    for i, (current_feature, current_title) in enumerate(zip(all_features, all_titles)):
        feature_values = np.array([features[current_feature] for features in all_data])

        mean_value = np.mean(feature_values)
        std_value = np.std(feature_values)
        normalized_feature_values = (feature_values - mean_value) / std_value

        indices = np.argsort(normalized_feature_values)
        sorted_feature_values = normalized_feature_values[indices]
        sorted_nn_output = average_nn_output[indices]

        num_parts = 50
        average_feature_values = []
        average_output_values = []
        output_sem_values = []

        for j in range(num_parts):
            index_start = j * len(sorted_feature_values) // num_parts
            index_end = (j + 1) * len(sorted_feature_values) // num_parts

            part_feature_values = sorted_feature_values[index_start:index_end]
            part_nn_output = sorted_nn_output[index_start:index_end]

            average_feature_values.append(np.mean(part_feature_values))
            average_output_values.append(np.mean(part_nn_output))
            output_sem_values.append(np.std(part_nn_output, ddof=1) / np.sqrt(len(part_nn_output)))

        axes[i].errorbar(
            average_feature_values,
            average_output_values,
            yerr=output_sem_values,
            fmt='o',
            capsize=5,
            color='blue',
            label='All answers',
            alpha=0.8
        )

        # Weighted least squares on the 50 binned means, weighted by 1/SEM: bins have
        # unequal precision, so this gives less influence to noisier bins rather than
        # treating all 50 bins as equally reliable (consistent with the fit used in
        # Fig4_features_interest_suggestion.py).
        bin_weights = 1.0 / np.array(output_sem_values)
        slope, intercept = np.polyfit(average_feature_values, average_output_values, 1, w=bin_weights)
        fit_x = np.array(average_feature_values)
        fit_y = slope * fit_x + intercept

        axes[i].plot(fit_x, fit_y, color='grey', linestyle='--', linewidth=2, label='Linear fit (all answers)')

        axes[i].set_title(current_title, fontsize=24)
        axes[i].grid(True)
        axes[i].tick_params(axis='x', labelsize=21)
        axes[i].tick_params(axis='y', labelsize=21, labelleft=True)

        axes[i].set_xlim(x_axis_ranges[i])

        axes[i].set_ylim(-0.05, 1.50)
        axes[i].set_yticks(np.arange(0.0, 1.51, 0.25))

        if i == 3:
            axes[i].legend(fontsize=19)

        for spine in axes[i].spines.values():
            spine.set_linewidth(1.6)

    fig.text(0.5, 0.02, 'Normalized Feature Values', ha='center', va='center', fontsize=26)
    fig.text(0.015, 0.5, 'Unnormalized prediction of high-impact of concept pair',
              ha='center', va='center', rotation='vertical', fontsize=26)

    plt.tight_layout(rect=[0.035, 0.04, 1, 1])
    fig.subplots_adjust(hspace=0.35, wspace=0.16)

    plt.savefig(os.path.join('Figures', 'Fig4_feature_impact_score_SI.png'), format='png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
