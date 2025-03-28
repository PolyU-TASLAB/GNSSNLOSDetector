# GNSS NLOS Detector

## Overview
This project implements a Non-Line-of-Sight (NLOS) detector for Global Navigation Satellite System (GNSS) signals. The detector helps identify when satellite signals are being received through reflections or obstructions rather than directly, which can significantly improve positioning accuracy in challenging environments like urban canyons.

## Features
**-Learning signal characteristics:** SNR, Elevation Angle, Azimuth Angle, Satellite Constellation.

**-Synthetic Dataset:** Simulates urban, suburban, and open-sky environments with realistic GPS signal patterns.

**-Machine Learning to detect NLOS:** Real-time NLOS detection using machine learning algorithms.

**-Support for multiple GNSS constellations:** GPS, GALILEO, GLONASS, BeiDou.

**-Signal quality analysis and visualization tools**

**-Performance metrics for detection accuracy**

## Installation
### Dependencies
- Python 3.7+
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `graphviz`


Install dependencies via `pip`:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy graphviz
```
## Walkthrough
**1. Generate Synthetic Dataset (Exercise 0):**
```bash
python3 GenDataset.py
```
Run GenDataset.py to generate synthetic data. You can see over 1000 samples in the gps_nlos_dataset.csv file.

***Data Exploration (Exercise 1):***
Visualize feature distributions, correlations, and class imbalances using pair plots, histograms, and heatmaps.

Preprocess data (one-hot encoding, train-test split, scaling).

![SNR vs Elevation](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/snr_vs_elevation.png)
![Satellite_sky_distribution](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/satellite_sky_distribution.png)
![feature_distribution](/GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/feature_distributions.png)
![feature_pairplot](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/feature_pairplot.png)
![correlation_matrix](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/correlation_matrix.png)
![3d_signal_features](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/3d_signal_features.png)

**Model Implementation (Exercise 2-5):**

***Exercise 2: Implement Linear Regression Model***

Train and evaluate models:

```bash
python3 LinearRegress.py
```

Implement linear regression model for NLOS classification, evaluate the model's performanceusing mses, r2 scores, accuracy and confusion matrix. The coefficients of the linear regression model can be used to understand how each feature contributes to NLOS classification. They are analyzed and visualized in the end.

![linear_regression_coefficients](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/linear_regression_coefficients.png)

![linear_regession_predictions](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/linear_regression_predictions.png)

![linear_regression_confusion_matrix](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/linear_regression_confusion_matrix.png)

![linear_regression_summary](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/linear_regression_summary.png)



***Exercise 3: Implement Logistic Regression Model***

```bash
python3 LogisticReg.py
```

Implement Logistic regression model for NLOS classification, evaluate the model's performanceusing mses, r2 scores, accuracy and confusion matrix. The coefficients of the linear regression model can be used to understand how each feature contributes to NLOS classification. They are analyzed and visualized in the end.

![logistic_regression_summary](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/logistic_regression_summary.png)

![logistic_regression_confusion_matrix](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/logistic_regression_confusion_matrix.png)

![logistic_regression_roc](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/logistic_regression_roc.png)

![logistic_regression_recision_recall](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/logistic_regression_precision_recall.png)

![logisitic_regression_probability_distribution](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/logistic_regression_probability_distribution.png)

![logistic_regression_odds_ratio](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/logistic_regression_odds_ratios.png)

![logistic_regression_coefficients](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/logistic_regression_coefficients.png)

![logistic_regression_boundary](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/logistic_regression_boundary.png)

***Exercise 4: Implement Decision Tree***

```bash
python3 DecisionTree.py
```

![decision_tree_visualization](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/decision_tree_visualization.png)

![decision_tree_importantance](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/decision_tree_feature_importance.png)

Implement Decision tree for NLOS classification, evaluate the model's performanceusing mses, r2 scores, accuracy and confusion matrix. The coefficients of the linear regression model can be used to understand how each feature contributes to NLOS classification. They are analyzed and visualized in the end.
![decision_tree_depth_analysis](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/decision_tree_depth_analysis.png)
![decision_tree_confusion_matrix](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/decision_tree_confusion_matrix.png)

![decision_tree_summary_table](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/decision_tree_summary.png)

***Exercise 5: Implement SVM (Support Vector Machines) & Exercise 6: Model Comparision and Real-World Application***

```bash
python3 Comparison.py
```

![svm_model_comparison](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/model_comparison.png)


Implement SVM classifier for NLOS detection. Explore the different kernel functions and evaluate performance of SVM kernels. In the end, tune the hyperparameters.

![svm_model_comparison_roc](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/model_comparison_roc.png)

![path_nlos_prediction](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/path_nlos_prediction.png)

![nlos_skyplot](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/nlos_skyplot.png)

![position_improvement](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/positioning_improvement.png)

![position_error_distribution](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/position_error_distribution.png)

![application_summary_table](GPS%20NLOS%20Signal%20Identification%20Using%20Machine%20Learning/Code/figures/application_summary.png)






