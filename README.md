# SciMuse

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2405.17044-b31b1b.svg)](https://arxiv.org/abs/2405.17044)

### How interesting are AI-generated research ideas to experienced human researchers, and how can we improve their quality?


📖 <u> Read our paper here: </u>\
[**Interesting Scientific Idea Generation Using Knowledge Graphs and LLMs: Evaluations with 100 Research Group Leaders**](https://arxiv.org/abs/2405.17044)\
*[Xuemei Gu](mailto:xuemei.gu@mpl.mpg.de), [Mario Krenn](mailto:mario.krenn@mpl.mpg.de)*

<img src="figures/scimuse.jpeg" alt="workflow" width="900"/>

> [!NOTE]\
> Full Dynamic Knowledge Graph can be downloaded at [10.5281/zenodo.13900962](https://doi.org/10.5281/zenodo.13900962)  

<pre>
.
├── <a href="https://github.com/artificial-scientist-lab/SciMuse/tree/main/data">data</a>                                      # Directory containing datasets
│   ├── full_concepts.txt                     # Full concept list
│   ├── all_evaluation_data.pkl               # Human evaluation dataset
│   ├── full_data_ML.pkl                      # Dataset for supervised neural networks (from create_full_data_ML_pkl.py)
│   ├── full_data_gpt35.pkl                   # Dataset for GPT-3.5 (from create_full_data_gpt_pkl.py)
│   ├── full_data_gpt4o.pkl                   # Dataset for GPT-4o (from create_full_data_gpt_pkl.py)
│   ├── full_data_gpt4omini.pkl               # Dataset for GPT-4omini
│   ├── full_data_DT_fixed_params.pkl         # Dataset for Decision tree
│   ├── elo_data_gpt35.pkl                    # ELO ranking data for GPT-3.5 (from create_full_data_gpt_pkl.py)
│   ├── elo_data_gpt4o.pkl                    # ELO ranking data for GPT-4o (from create_full_data_gpt_pkl.py)
│   ├── combined_ELO_results_35.txt           # ELO results for GPT-3.5
│   ├── combined_ELO_results_4omini.txt       # ELO results for GPT-4omini
│   └── combined_ELO_results_4o.txt           # ELO results for GPT-4o
│
├── <a href="https://github.com/artificial-scientist-lab/SciMuse/tree/main/figures">figures</a>                                   # Directory for storing generated figures
│
├── create_fig3.py                            # Analysis of interest levels vs. knowledge graph features (for Fig. 3)
├── create_full_data_ML_pkl.py                # Code for generating supervised ML dataset (full_data_ML.pkl)
├── create_full_data_gpt_pkl.py               # Code for generating GPT datasets (full_data_gpt35.pkl, full_data_gpt4o.pkl, etc.)
├── create_fig4.py                            # Predicting scientific interest and generating Fig. 4
├── create_figs_withTree.py                   # Predicting scientific interest and generating Fig4 with Decision tree in the SI
│
└── Fig_AUC_over_time.py                      # Zero-shot ranking of research suggestions by LLMs (for Fig. 6)
</pre>
