# Hybrid Intrusion Detection Model 2.0

## Overview
This project builds upon a previous research article that proposed a hybrid intrusion detection system (IDS) for cloud environments. The goal of the research was to enhance intrusion detection capabilities by combining multiple machine learning models. In this project, we extend the original work by incorporating more datasets, experimenting with various machine learning models, optimizing model performance, and introducing ensemble models to improve the efficacy of intrusion detection.

The original paper utilized a combination of signature-based and anomaly-based detection methods, tested on the CIC-IDS2017, NSL-KDD, SDN, and UNSW-NB15 datasets. This project aims to refine the hybrid IDS model by assessing different algorithms and evaluating them through metrics like accuracy, precision, recall, and F1 score. By leveraging ensemble models, we intend to enhance detection accuracy and reduce false positives.

## Objectives
1. Expand the dataset scope to include additional datasets to test the robustness of the models.
2. Evaluate and optimize a range of machine learning models for intrusion detection.
3. Apply and compare ensemble models to create a more resilient intrusion detection system.
4. Improve model optimization through advanced hyperparameter tuning and ensemble techniques.

## Datasets Used
- **CIC-IDS2017**: Dataset simulating real-world network traffic scenarios, including various attack types.
- **NSL-KDD**: Improved version of the KDD Cup 99 dataset, addressing issues like redundant records.
- **SDN**: Dataset representing traffic from Software-Defined Networking environments.
- **UNSW-NB15**: Comprehensive dataset with a wide variety of attack categories.

## Methodology
- **Model Selection**: AdaBoost, Decision Tree, Random Forest, Gradient Boosting, Extremely Randomized Trees, and Neural Networks were used. These models were selected due to their past performance in intrusion detection tasks.
- **Model Optimization**: Models were optimized using Hyperband, Grid Search CV, Gradient-Based Optimization, Optuna, and Bayesian techniques.
- **Ensemble Techniques**: Techniques like Hard Voting, Soft Voting, Bayesian Model Averaging, and Weighted Averaging were employed to form an ensemble model that aims to leverage the strengths of individual models.
- **Performance Metrics**: Accuracy, precision, recall, F1 score, AUC, and confusion matrices were used to assess model performance.

## Results Summary
The results indicate that ensemble models significantly improved the performance of individual machine learning models. Specifically, the **Weighted Average** ensemble technique showed the best results across datasets, offering the highest accuracy and lowest false positive rate.

- **Individual Model Performance**: Random Forest and Decision Tree consistently outperformed other models in most datasets, while Neural Networks required further tuning.
- **Ensemble Model Performance**: Hard Voting and Weighted Average methods provided the highest performance for different datasets, demonstrating the robustness of combining models.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/intrusion-detection-enhancements.git
   ```
2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Data Preprocessing**: Use the preprocessing scripts in the `src/preprocessing.py` file to clean and prepare the datasets.
2. **Model Training**: Run `src/modeling.py` to train individual models. Use the `optimization.py` script for hyperparameter tuning.
3. **Ensemble Models**: Use `src/ensemble.py` to combine and evaluate the ensemble models.
4. **Evaluation**: Run `src/evaluate.py` to generate performance metrics and visualize model performance.

## Future Work
1. **Computational Efficiency**: Further optimize the models for real-time intrusion detection.
2. **Scalability**: Evaluate the performance of the hybrid model in big data scenarios.
3. **Deep Learning Integration**: Explore incorporating deep learning architectures for enhanced detection capabilities.
4. **Real-Time Threat Intelligence**: Integrate real-time threat feeds to improve model adaptability.

## References
- [Original Research Paper](https://doi.org/10.1007/s11277-022-10063-y)
- Various other references and literature that contributed to the development of the project.

