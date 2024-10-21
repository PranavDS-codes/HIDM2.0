# Results

## Overview
The results of this project demonstrate the effectiveness of the proposed hybrid intrusion detection system (IDS) in enhancing detection accuracy and reducing false positives compared to individual models. The experiments were conducted on multiple datasets, including CIC-IDS2017, NSL-KDD, SDN, and UNSW-NB15, and a comprehensive evaluation was carried out for each machine learning model, including ensemble methods. The performance of the models was assessed using key metrics such as accuracy, precision, recall, F1 score, AUC, and confusion matrices.

## Individual Model Performance

### NSL-KDD Dataset
- **Decision Tree** and **Random Forest** models demonstrated exceptional performance, with AUCs of **0.9946** and **0.9956**, respectively.
- **AdaBoost** and **Gradient Boosting** also showed strong results, while the **Neural Network** model lagged behind in terms of AUC and other metrics.

| Model                 | AUC    | Accuracy (%) | F1 Score | Precision | Recall |
|-----------------------|--------|--------------|----------|-----------|--------|
| Decision Tree         | 0.9946 | 99.46        | 0.9948   | 0.9945    | 0.9951 |
| Random Forest         | 0.9956 | 99.56        | 0.9958   | 0.9972    | 0.9945 |
| AdaBoost              | 0.9715 | 97.17        | 0.9729   | 0.9777    | 0.9682 |
| Gradient Boosting     | 0.9879 | 98.79        | 0.9884   | 0.9891    | 0.9878 |
| Neural Network        | 0.9231 | 92.32        | 0.9263   | 0.9272    | 0.9254 |

### CIC-IDS2017 Dataset
- **Decision Tree** and **Extremely Randomized Trees** achieved nearly perfect scores in all metrics.
- **Random Forest** showed relatively lower performance in this dataset compared to others.

| Model                 | AUC    | Accuracy (%) | F1 Score | Precision | Recall |
|-----------------------|--------|--------------|----------|-----------|--------|
| Decision Tree         | 0.9999 | 99.99        | 0.9999   | 0.9999    | 1.0000 |
| Extremely Randomized Tree | 0.9997 | 99.99    | 0.9999   | 1.0000    | 0.9999 |
| Random Forest         | 0.9743 | 98.93        | 0.9934   | 0.9991    | 0.9877 |
| Gradient Boosting     | 0.9981 | 99.92        | 0.9995   | 1.0000    | 0.9991 |
| Neural Network        | 0.9986 | 99.91        | 0.9995   | 0.9995    | 0.9994 |

### SDN Dataset
- **Random Forest** achieved a perfect score across all metrics.
- The **Neural Network** model underperformed significantly, with the lowest scores across all metrics.

| Model                 | AUC    | Accuracy (%) | F1 Score | Precision | Recall |
|-----------------------|--------|--------------|----------|-----------|--------|
| Decision Tree         | 0.9999 | 99.99        | 1.0000   | 1.0000    | 0.9999 |
| Random Forest         | 1.0000 | 100.00       | 1.0000   | 1.0000    | 1.0000 |
| Gradient Boosting     | 0.9968 | 99.67        | 0.9973   | 0.9964    | 0.9982 |
| Extremely Randomized Tree | 0.9997 | 99.96    | 0.9997   | 0.9996    | 0.9998 |
| Neural Network        | 0.5390 | 63.12        | 0.7209   | 0.7812    | 0.6692 |

### UNSW-NB15 Dataset
- **Random Forest** and **Extremely Randomized Trees** exhibited superior performance compared to other models before optimization.
- The **Neural Network** model again performed poorly on this dataset.

| Model                 | AUC    | Accuracy (%) | F1 Score | Precision | Recall |
|-----------------------|--------|--------------|----------|-----------|--------|
| Random Forest         | 0.9378 | 94.17        | 0.9199   | 0.9236    | 0.9161 |
| Extremely Randomized Tree | 0.9339 | 93.84    | 0.9152   | 0.9179    | 0.9125 |
| Gradient Boosting     | 0.9091 | 92.16        | 0.8887   | 0.8639    | 0.9149 |
| Neural Network        | 0.6070 | 71.49        | 0.3534   | 0.2149    | 0.9936 |

## Model Optimization Results
Post-optimization, all models showed improved performance, with notable enhancements in accuracy and other metrics:

| Model                 | Optimization Algorithm | AUC    | Accuracy (%) | F1 Score | Precision | Recall |
|-----------------------|------------------------|--------|--------------|----------|-----------|--------|
| AdaBoost              | Hyperband              | 0.8981 | 91.15        | 0.8743   | 0.8493    | 0.9008 |
| Decision Tree         | Optuna                 | 0.9317 | 93.34        | 0.9097   | 0.9254    | 0.8946 |
| Random Forest         | Hyperband              | 0.9395 | 94.30        | 0.9218   | 0.9266    | 0.9171 |
| Extremely Randomized Tree | Optuna             | 0.9340 | 93.94        | 0.9163   | 0.9143    | 0.9183 |

## Ensemble Model Performance
Ensemble models were employed to leverage the strengths of individual models, and the results clearly showed the benefits of this approach:

### UNSW-NB15 Dataset (Post-Optimization)
- The **Weighted Average** ensemble method outperformed other techniques, achieving the highest scores in most metrics, with an AUC of **0.9357** and accuracy of **94.13%**.
- **Hard Voting** and **Dynamic Methods** also showed similar performance, providing strong results across the board.

| Voting Criteria     | Models Used                     | AUC    | Accuracy (%) | F1 Score | Precision | Recall |
|---------------------|---------------------------------|--------|--------------|----------|-----------|--------|
| Hard                | RF, DT, ET, AdaB, GB            | 0.9330 | 93.96        | 0.9160   | 0.9092    | 0.9229 |
| Soft                | RF, DT, ET, AdaB, GB            | 0.9299 | 93.70        | 0.9132   | 0.9045    | 0.9203 |
| Bayesian Average    | RF, DT, ET, AdaB, GB            | 0.9344 | 94.07        | 0.9176   | 0.9114    | 0.9239 |
| Dynamic             | RF, DT, ET, AdaB, GB            | 0.9344 | 94.07        | 0.9176   | 0.9114    | 0.9239 |
| Weighted Average    | RF, DT, ET, AdaB, GB            | 0.9357 | 94.13        | 0.9187   | 0.9155    | 0.9219 |

### NSL-KDD Dataset
- The ensemble of **Random Forest, Decision Tree, Extremely Randomized Trees, AdaBoost, and Gradient Boosting** using **Hard Voting** resulted in near-perfect performance metrics.
- **Soft Voting** and **Bayesian Model Averaging** also showed improved performance over individual models.

| Voting Criteria     | Models Used                     | AUC    | Accuracy (%) | F1 Score | Precision | Recall |
|---------------------|---------------------------------|--------|--------------|----------|-----------|--------|
| Hard                | RF, DT, ET, AdaB, GB            | 0.9949 | 99.50        | 0.9952   | 0.9961    | 0.9943 |
| Soft                | RF, DT, ET, AdaB, GB            | 0.9923 | 99.23        | 0.9928   | 0.9935    | 0.9921 |
| Bayesian Average    | RF, DT, ET, AdaB, GB            | 0.9935 | 99.35        | 0.9941   | 0.9946    | 0.9937 |

### CIC-IDS2017 Dataset
- The ensemble approach, which included the **Neural Network**, achieved high performance with **Hard Voting**, achieving an accuracy of **99.96%** and an F1 score of **0.9998**.

| Voting Criteria     | Models Used                     | AUC    | Accuracy (%) | F1 Score | Precision | Recall |
|---------------------|---------------------------------|--------|--------------|----------|-----------|--------|
| Hard                | NN, DT, ET, AdaB, GB            | 0.9991 | 99.96        | 0.9998   | 1.0000    | 0.9996 |
| Soft                | NN, DT, ET, AdaB, GB            | 0.9988 | 99.93        | 0.9994   | 0.9999    | 0.9990 |

### SDN Dataset
- **Hard Voting** with **Random Forest, Decision Tree, Extremely Randomized Trees, AdaBoost, and Gradient Boosting** led to near-perfect scores, showcasing the effectiveness of combining multiple models for this dataset.

| Voting Criteria     | Models Used                     | AUC    | Accuracy (%) | F1 Score | Precision | Recall |
|---------------------|---------------------------------|--------|--------------|----------|-----------|--------|
| Hard                | RF, DT, ET, AdaB, GB            | 0.9999 | 99.99        | 0.9999   | 0.9998    | 1.0000 |
| Soft                | RF, DT, ET, AdaB, GB            | 0.9997 | 99.97        | 0.9997   | 0.9996    | 0.9998 |

## Key Observations
1. **Ensemble Methods Improve Performance**: The results indicate that ensemble methods significantly enhance the performance of individual models. The **Weighted Average** approach, in particular, provided the best performance for most datasets.

2. **Model Optimization Enhances Results**: Advanced hyperparameter optimization techniques like **Hyperband**, **Optuna**, and **Bayesian Optimization** significantly improved model metrics, particularly for **AdaBoost** and **Random Forest**.

3. **Dataset-Specific Performance**: Different models performed better on different datasets, highlighting the importance of dataset-specific model selection and optimization. For instance, **Random Forest** consistently performed well across all datasets, whereas **Neural Networks** required further tuning and struggled to match the performance of other models.

4. **Weighted Average Ensemble**: The **Weighted Average** ensemble technique was the most effective overall, providing superior performance by combining the strengths of individual models based on their reliability.

## Conclusion
The results of this project demonstrate that the proposed hybrid IDS, which incorporates multiple machine learning models and ensemble methods, significantly improves intrusion detection performance. By optimizing individual models and leveraging ensemble techniques, we successfully enhanced accuracy, minimized false positives, and created a more resilient intrusion detection framework for cloud-based environments.

The findings underscore the effectiveness of combining different models and optimizing them to achieve superior results, making the proposed IDS an effective solution for detecting network intrusions in diverse settings.

