# CdWise-ML-Framework-Scripts

# CdWise-ML-Framework: A Machine Learning Framework for Health Risk Prediction of Cadmium in Vegetables

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official source code and machine learning models for the research paper:

> **"The development of the 'Cd-Wise' app for human health risk assessment, dietary recommendations, and planting structure adjustment of cadmium in chinese vegetables based on machine learning"**  
> *Xingang Liu, Jie liu, Feng Li, et al.*  
> *(Journal Name, Year)* - [Link to Paper, when available]

Institution: This project was developed at the:
a Hunan Provincial University Key Laboratory for Environmental and Ecological Health, Hunan Provincial University Key Laboratory for Environmental Behavior and Control Principle of New Pollutants, College of Environment and Resources, Xiangtan University, Xiangtan 411105, China

b National Center for Applied Mathematics in Hunan, the Key Laboratory of Intelligent Computing and Information Processing of the Ministry of Education, at School of Mathematics and Computational Science, Xiangtan University, Xiangtan 411105, China


This framework is designed to predict the human health risk (Target Hazard Quotient, THQ) associated with dietary exposure to cadmium (Cd) in vegetables, based on a comprehensive, multidimensional dataset (CVCCD).

## Overview

The core of this research is a sophisticated machine learning framework developed to model the complex relationships between environmental factors, vegetable characteristics, and population-specific health risks. To effectively mine predictive information from our 32-dimensional dataset, we developed an integrated learning framework that addresses two primary methodological challenges:

1.  **Model Selection and Robustness:** To handle the complex, non-linear interactions within the high-dimensional data, our architecture strategically combines a diverse suite of algorithms representing the major classes of machine learning:
    *   **Bagging:** Random Forest (RF)
    *   **Gradient Boosting:** GBDT, XGBoost, LightGBM
    *   **Kernel-based:** Support Vector Machine (SVM)
    *   **Deep Learning:** Convolutional Neural Network (CNN)

2.  **Performance Optimization:** To maximize the predictive accuracy of this data-intensive approach, we implemented a sophisticated hyperparameter optimization protocol that combines **Bayesian Optimization** with **10-fold cross-validation**.

This repository provides the Python scripts to replicate the model training, evaluation, and analysis presented in our paper.

## Features

*   **Comprehensive Data Preprocessing:** Scripts for cleaning, encoding categorical features, and preparing datasets for both tree-based and non-tree-based models.
*   **Multi-Target Modeling:** Independent models are trained for four distinct population groups: Urban Male, Urban Female, Rural Male, and Rural Female.
*   **Advanced Hyperparameter Tuning:** Implementation of Bayesian Optimization to efficiently find the best hyperparameters for RF, GBDT, XGBoost, and LightGBM.
*   **Multi-Model Evaluation:** A complete pipeline for training and evaluating six different ML/DL models (CNN, RF, SVM, XGBoost, GBDT, LightGBM).
*   **Model Interpretation:** SHAP (SHapley Additive exPlanations) analysis to interpret model predictions and identify key risk drivers.
*   **Visualization:** Code to generate convergence curves, model performance comparisons, confusion matrices, and ROC curves.
*   **Reproducibility:** Pre-trained models, scalers, and encoders are saved to ensure full reproducibility of our results.

## Repository Structure

```
.
├── models/                  # Directory to save trained models and scalers
│   ├── urban_male_lgbm_model.pkl
│   ├── category_mappings.pkl
│   └── ...
├── results/                 # Directory for output figures and data
│   ├── Urban_Male_Model_model_evaluation.pdf
│   ├── all_population_model_confusion_matrices.png
│   └── ...
├── data/                    # Placeholder for the dataset
│   └── EN中国蔬菜镉含量数据库_all9.xlsx  (Note: Dataset not included, please refer to the paper's data availability statement)
├── health_risk_prediction.py  # Main Python script for the entire workflow
├── requirements.txt         # Required Python libraries
└── README.md                # This file
```

## Getting Started

### Prerequisites

This project is developed in Python 3.9. Ensure you have Python installed. You will also need to install the required libraries.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/YourUsername/CdWise-ML-Framework.git
    cd CdWise-ML-Framework
    ```

2.  Install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Place the Dataset:** Obtain the dataset (`EN中国蔬菜镉含量数据库_all9.xlsx`) as described in our paper's data availability statement and place it in the `data/` directory.

2.  **Update File Path:** Open `health_risk_prediction.py` and update the path to the dataset in the following line:
    ```python
    # Line 43 (approx.)
    data = pd.read_excel(r'./data/EN中国蔬菜镉含量数据库_all9.xlsx')
    ```

3.  **Run the Script:** Execute the main script from the terminal. This will perform all steps: data preprocessing, model training with hyperparameter optimization, evaluation, and generation of results (figures and data files) in the `results/` directory.
    ```bash
    python health_risk_prediction.py
    ```
    The script will log its progress to the console. The entire process may take some time, especially the Bayesian optimization part.

## Citation

If you use this code or the associated models in your research, please cite our paper:

```bibtex
@article{Liu_2023_CdWise,
  author    = {Liu, Xingang and Liu, Jie and Li, Feng and Yi, Shengwei and Han, Guosheng and Zhu, Lizhong and Wu, Yujun and Mei, Yancheng and Chen, Jie},
  title     = {The development of the "Cd-Wise" app for human health risk assessment, dietary recommendations, and planting structure adjustment of cadmium in chinese vegetables based on machine learning},
  journal   = {Journal Name},
  year      = {2023},
  volume    = {XX},
  number    = {X},
  pages     = {XXX--XXX},
  doi       = {DOI link here}
}
```
*(Please update the BibTeX entry with the final publication details.)*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
