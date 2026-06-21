Here is the revised README file, strictly following the structure, style, and content volume of your provided reference, while accurately reflecting the specific "No-Leakage" methodology found in your script and incorporating the required institutional and file details.

---

# CdWise-ML-Framework (No-Leakage Version): A Machine Learning Framework for Health Risk Prediction of Cadmium in Vegetables

This repository contains the official source code and machine learning models for the research paper:

> **"The development of the 'Cd-Wise' app for human health risk assessment, dietary recommendations, and planting structure adjustment of cadmium in chinese vegetables based on machine learning"**
> *Xingang Liu, Jie Liu, Feng Li, et al.*
> *(Journal Name, Year)* - [Link to Paper, when available]

**Institutional Affiliation:** This project was developed at the:

* Hunan Provincial University Key Laboratory for Environmental and Ecological Health
* Hunan Provincial University Key Laboratory for Environmental Behavior and Control Principle of New Pollutants
* College of Environment and Resources, Xiangtan University, Xiangtan 411105, China

This framework is designed to predict the human health risk (Target Hazard Quotient, THQ) associated with dietary exposure to cadmium (Cd) in vegetables. Specifically, this branch features a **"No-Leakage"** protocol, which strictly excludes post-harvest variables (such as Vegetable Cd concentrations, BCF, and SCC) to enable true pre-harvest risk forecasting based purely on environmental, geographical, and demographic data.

## Overview

The core of this research is a sophisticated machine learning framework developed to model the complex relationships between environmental factors, vegetable characteristics, and population-specific health risks without relying on target-leaking features. To effectively mine predictive information from our dataset, we developed an integrated learning framework that addresses two primary methodological challenges:

1. **Model Selection and Robustness:** To handle the complex, non-linear interactions within the high-dimensional environmental and demographic data, our architecture strategically combines a diverse suite of algorithms representing the major classes of machine learning:
* **Bagging:** Random Forest (RF)
* **Gradient Boosting:** GBDT, XGBoost, LightGBM
* **Kernel-based:** Support Vector Machine (SVM)
* **Deep Learning:** Convolutional Neural Network (CNN)


2. **Performance Optimization:** To maximize the predictive accuracy of this data-intensive approach, we implemented a sophisticated hyperparameter optimization protocol that utilizes **Bayesian Optimization** (via `skopt`) to fine-tune the tree-based models, preventing overfitting by strictly evaluating on validation sets.

This repository provides the Python scripts to replicate the model training, evaluation, and interpretability analysis presented in our paper.

## Features

* **Strict No-Leakage Protocol:** Automatically drops target-leaking features (`Vegetable Cadmium (mg/kg)`, `BCF`, `SCC`) to construct a strictly honest pre-harvest prediction modeling dataset.
* **Multi-Target Modeling:** Independent models are trained and evaluated for four distinct population groups: Urban Male, Urban Female, Rural Male, and Rural Female.
* **Advanced Hyperparameter Tuning:** Implementation of Bayesian Optimization to efficiently find the best hyperparameters for RF, GBDT, XGBoost, and LightGBM.
* **Multi-Model Evaluation:** A complete pipeline for training and evaluating six different ML/DL models, automatically calculating AUC, ACC, Sensitivity (SE), and F1 scores.
* **Model Interpretation:** Comprehensive SHAP (SHapley Additive exPlanations) analysis (beeswarm, heatmap, waterfall, force, and decision plots) for the LightGBM model to identify key environmental risk drivers.
* **Automated Visualization & Export:** The script automatically generates an output directory containing convergence curves, model performance comparisons, ROC curves, confusion matrices, and the complete "honest" datasets.

## Repository Structure

```text
.
├── noleak_outputs/          # Auto-generated directory for output figures and data
│   ├── all_population_model_confusion_matrices.pdf
│   ├── test_acc_auc_table_NOLEAK.xlsx
│   ├── Urban_Male_model_evaluation.pdf
│   └── ...
├── data/                    # Recommended directory for the dataset
│   └── EN中国蔬菜镉含量数据库_all9_noleak.xlsx  (Note: Dataset must be placed here)
├── p08_noleak_full_outputs.py # Main Python script for the entire no-leakage workflow
├── requirements.txt         # Required Python libraries
└── README.md                # This file

```

## Getting Started

### Prerequisites

This project is developed in Python 3.8+. Ensure you have Python installed. You will also need to install the required libraries.

### Installation

1. Clone the repository:
```bash

```



git clone https://github.com/YourUsername/CdWise-ML-Framework.git
cd CdWise-ML-Framework

```

2.  Install the required packages using pip:
    ```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch xgboost lightgbm shap scikit-optimize openpyxl

```

### Usage

1. **Place the Dataset:** Obtain the dataset `EN中国蔬菜镉含量数据库_all9_noleak.xlsx` as described in our paper's data availability statement and ensure it is placed in the correct directory.
2. **Update File Path:** Open `p08_noleak_full_outputs.py` and update the `DATA_PATH` variable to point to your dataset if it is not in the same directory:
```python

```



# Line 51 (approx.)

DATA_PATH = r'./data/EN中国蔬菜镉含量数据库_all9_noleak.xlsx'

```

3.  **Run the Script:** Execute the main script from the terminal. This will perform all steps: data loading, leakage-feature dropping, model training with Bayesian optimization, evaluation, and the generation of all visual results and excel summaries into the `noleak_outputs/` directory.
    ```bash
python p08_noleak_full_outputs.py

```

```
The script will log its progress to the console. The entire process may take some time depending on your hardware, especially during the CNN training and Bayesian optimization phases.

```

## Citation

If you use this code or the associated models in your research, please cite our paper:

```bibtex
@article{Liu_202X_CdWise,
  author    = {Liu, Xingang and Liu, Jie and Li, Feng and Yi, Shengwei and Han, Guosheng and Zhu, Lizhong and Wu, Yujun and Mei, Yancheng and Chen, Jie},
  title     = {The development of the "Cd-Wise" app for human health risk assessment, dietary recommendations, and planting structure adjustment of cadmium in chinese vegetables based on machine learning},
  journal   = {Journal Name},
  year      = {202X},
  volume    = {XX},
  number    = {X},
  pages     = {XXX--XXX},
  doi       = {DOI link here}
}

```

*(Please update the BibTeX entry with the final publication details.)*

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
