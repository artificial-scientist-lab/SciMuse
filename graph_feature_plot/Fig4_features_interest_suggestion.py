import os
import csv
import numpy as np
import matplotlib.pyplot as plt

DATASET_FILE = 'evaluation_feature_dataset.csv'


def load_survey_data(dataset_file, suggestiontype, institutetype, authortype):
    total_data = {"clarity": [], "interest": [], "features": []}

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
                total_data["clarity"].append(clarity)
                total_data["interest"].append(interest)
                total_data["features"].append([float(v) for v in row[5:]])

    return total_data


def rankdata_avg(x):
    """Average ranks (1-indexed), ties get the mean of the ranks they span."""
    x = np.asarray(x)
    order = x.argsort(kind='mergesort')
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    sorted_x = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def pearson_r(x, y):
    return np.corrcoef(x, y)[0, 1]


def spearman_r(x, y):
    return pearson_r(rankdata_avg(x), rankdata_avg(y))


def print_fit_stats_table(fit_stats):
    """Print binned-fit and idea-level correlation stats, one block per feature.

    Binned R^2 is the fit quality of the plotted line to the 50 bin-means; it is
    inflated relative to idea-level predictability because binning averages away
    within-bin noise. Raw Pearson r^2 / Spearman rho are computed on the individual,
    unbinned ratings and are the numbers that should be reported/cited as predictive
    strength; the binned R^2 should only be described as a fit to bin-means.
    """
    for s in fit_stats:
        print(f"name: {s['title']}")
        print(f"slope: {s['slope']:.4f} +- {s['slope_err']:.4f}")
        print(f"intercept: {s['intercept']:.4f} +- {s['intercept_err']:.4f}")
        print(f"binned r2: {s['binned_r2']:.4f}")
        print(f"idea-level Spearman rho: {s['raw_rho']:+.4f}")
        print(f"idea-level Pearson r2: {s['raw_r2']:.4f}")
        print()


def select_top_n_percent(all_data, N):
    num_entries = len(all_data['features'])
    top_n_count = int(num_entries * (N / 100))

    impact_values = [(i, features[141]) for i, features in enumerate(all_data['features'])]
    sorted_by_impact = sorted(impact_values, key=lambda x: x[1], reverse=True)
    top_n_indices = [index for index, _ in sorted_by_impact[:top_n_count]]

    return {
        'clarity': [all_data['clarity'][i] for i in top_n_indices],
        'interest': [all_data['interest'][i] for i in top_n_indices],
        'features': [all_data['features'][i] for i in top_n_indices],
    }


def main():
    os.makedirs('Figures', exist_ok=True)
    authortype = ['nat', 'soc']
    institutetype = ['nat', 'same', 'soc']
    suggestiontype = ['random', 'semnet']

    all_features = [0, 14, 20, 26, 75, 87, 137, 143]
    all_titles = ['Degree of node A\n', 'PageRank of node A\n', 'Citation for node A\n', 'Total Citation for node A\n',
                  "Rank of 1-year citation increase\n for node B", 'Simpson similarity coefficient\nfor pair (A,B)',
                  'Total papers on concept A or B\n until two years ago, minimum count',
                  "Semantic distance\n"]

    color_map = {100: 'blue', 50: 'green', 25: 'red'}

    all_data = load_survey_data(DATASET_FILE, suggestiontype=suggestiontype, institutetype=institutetype, authortype=authortype)

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(28, 14))
    axes = axes.flatten()

    fit_stats = []

    for i, (curr_feature, curr_title) in enumerate(zip(all_features, all_titles)):
        all_feature_vals = np.array([ff[curr_feature] for ff in all_data['features']])
        mean_val = np.mean(all_feature_vals)
        std_val = np.std(all_feature_vals)

        raw_feature_vals_z = None
        raw_interest_data = None

        for percentage in [25, 50, 100]:
            all_data_top_n = select_top_n_percent(all_data, percentage)
            interest_data = np.array(all_data_top_n['interest'])
            feature_vals = np.array([ff[curr_feature] for ff in all_data_top_n['features']])

            feature_vals_z = (feature_vals - mean_val) / std_val

            if percentage == 100:
                # Raw, idea-level (unbinned) values used for the idea-level correlation below.
                raw_feature_vals_z = feature_vals_z
                raw_interest_data = interest_data

            indices = np.argsort(feature_vals_z)
            sorted_feature_vals = feature_vals_z[indices]
            sorted_interest_data = interest_data[indices]

            num_parts = 50
            avg_interest_data, std_interest_data, avg_feature_vals = [], [], []
            for j in range(num_parts):
                index_start = j * len(sorted_feature_vals) // num_parts
                index_end = (j + 1) * len(sorted_feature_vals) // num_parts if j != num_parts - 1 else len(sorted_feature_vals)
                part_interest_data = sorted_interest_data[index_start:index_end]
                avg_interest_data.append(np.mean(part_interest_data))
                std_interest_data.append(np.std(part_interest_data, ddof=1) / np.sqrt(len(part_interest_data)))
                avg_feature_vals.append(np.mean(sorted_feature_vals[index_start:index_end]))

            curr_label = 'All answers' if percentage == 100 else f'Top {percentage}% impact'
            axes[i].errorbar(avg_feature_vals, avg_interest_data, yerr=std_interest_data, fmt='o', capsize=5,
                              color=color_map[percentage], label=curr_label, alpha=0.8)

        # Weighted least squares: bins have unequal precision (different within-bin
        # variance), so weight each bin mean by 1/SEM the way the plotted error bars
        # already imply, rather than treating all 50 bins as equally reliable.
        bin_weights = 1.0 / np.array(std_interest_data)
        (slope, intercept), cov = np.polyfit(avg_feature_vals, avg_interest_data, 1, w=bin_weights, cov=True)
        slope_err, intercept_err = np.sqrt(np.diag(cov))

        # Binned fit quality (what the plotted dashed line achieves on the 50 bin-means)
        # vs. idea-level, unbinned correlation over all individual ratings. Binning
        # averages away within-bin noise, so the binned R^2 is not a measure of
        # idea-level predictability -- report both, and say so in any caption/table.
        binned_r2 = pearson_r(avg_feature_vals, avg_interest_data) ** 2
        raw_r2 = pearson_r(raw_feature_vals_z, raw_interest_data) ** 2
        raw_rho = spearman_r(raw_feature_vals_z, raw_interest_data)

        fit_stats.append({
            'title': ' '.join(curr_title.split()),
            'slope': slope, 'slope_err': slope_err,
            'intercept': intercept, 'intercept_err': intercept_err,
            'binned_r2': binned_r2, 'raw_r2': raw_r2, 'raw_rho': raw_rho,
            'n': len(raw_interest_data),
        })

        fit_line_linear = slope * np.array(avg_feature_vals) + intercept
        axes[i].plot(avg_feature_vals, fit_line_linear, 'grey', linestyle='--', linewidth=2, label='Linear Fit (all answers)')

        axes[i].set_title(curr_title, fontsize=24)
        axes[i].grid(True)
        if i == 3:
            axes[i].legend(fontsize=22)

        axes[i].tick_params(axis='x', labelsize=23)
        axes[i].set_ylim(1, 5)

        y_ticks = np.linspace(1, 5, 5)
        axes[i].set_yticks(y_ticks)
        axes[i].set_yticklabels(['{:.1f}'.format(y) for y in y_ticks], fontsize=23)

        for spine in axes[i].spines.values():
            spine.set_linewidth(1.6)

    print_fit_stats_table(fit_stats)

    fig.text(0.5, 0.02, 'Normalized Feature Values', ha='center', va='center', fontsize=26)
    fig.text(0.02, 0.5, 'Average Interest', ha='center', va='center', rotation='vertical', fontsize=26)

    plt.tight_layout(rect=[0.03, 0.03, 1, 1])
    fig.subplots_adjust(hspace=0.35, wspace=0.16)
    plt.savefig(os.path.join('Figures', 'Fig4_feature.png'), format='png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
