# Wearable Robotics Project: Binary Classifier for Gait EMG Signals

This project aims to design a binary classifier for gait Electromyography (EMG) signals using knee muscle data. Below are the steps and suggestions for improving the signal processing, feature extraction, classification, and future work to enhance performance, interpretability, and scalability.

## Project Overview

The goal is to classify EMG signals into two categories based on the knee muscle activity during gait. The following approach details the phases of the project:

1. **Signal Preprocessing and Feature Extraction**
2. **Binary Classification**
3. **Overfitting Detection & Dimensionality Reduction**
4. **K-Fold Cross Validation**
5. **Model Evaluation & Deployment**
6. **Interpretability and Explainability**
7. **Long-Term Monitoring & Continuous Learning**

## Phases

### Phase 1: Signal Preprocessing and Feature Extraction

#### Modifications:

- **Noise Reduction:** 
  - Apply bandpass filters (e.g., 20-500 Hz) to reduce noise and muscle artifacts.
  
- **Segmentation:** 
  - Segment the continuous EMG signal into smaller windows (e.g., 100ms or 200ms).

- **Normalization:** 
  - Normalize the EMG signal using Z-score normalization or min-max scaling.

- **Feature Engineering:** 
  - In addition to time-domain features (e.g., RMS, zero-crossings), add frequency-domain features (e.g., power spectral density) and time-frequency features (e.g., wavelet transform).

#### Future Scope:
- Investigate deep learning methods (e.g., CNN, LSTM) for automatic feature extraction and classification.

### Phase 2: Binary Classification Using All Features

#### Modifications:

- **Model Selection:** 
  - Compare multiple classifiers (Logistic Regression, SVM, Random Forest, Neural Networks) and evaluate using accuracy, precision, recall, and F1 score.

- **Handling Class Imbalance:** 
  - If the data is imbalanced, apply techniques like SMOTE or adjust the loss function using class weights.

- **Ensemble Learning:** 
  - Consider ensemble methods (e.g., Random Forest, AdaBoost, XGBoost) for improved performance.

#### Future Scope:
- Explore transfer learning with external gait datasets for better generalization.
- Use deep learning models (e.g., CNN, RNN) for automatic classification.

### Phase 3: Checking for Overfitting and Dimensionality Reduction

#### Modifications:

- **Regularization:** 
  - Use L1/L2 regularization (Lasso/Ridge regression) or dropout in neural networks to avoid overfitting.

- **Dimensionality Reduction:** 
  - Apply PCA or LDA for reducing the feature space. Alternatively, use t-SNE for visualization.

- **Cross-Validation:** 
  - Implement stratified k-fold cross-validation to maintain class proportions.

#### Future Scope:
- Investigate Autoencoders for unsupervised feature selection.
- Use SHAP or other feature importance techniques for model interpretation.

### Phase 4: K-Fold Cross Validation

#### Modifications:

- **Stratified K-Fold:** 
  - Ensure stratified k-fold to preserve class distribution in each fold.

- **Hyperparameter Tuning:** 
  - Use grid search or random search to optimize model hyperparameters.

#### Future Scope:
- As more data is collected, implement leave-one-subject-out cross-validation for evaluating model generalization.

---

### Phase 5: Model Evaluation and Deployment

- **Test Evaluation:** 
  - Evaluate the final model on a separate test dataset or real-world data to measure generalization.

- **Deployment:** 
  - Explore options for deploying the model in real-time wearable devices for continuous gait monitoring.

---

### Phase 6: Interpretability and Explainability

- **Explainability:** 
  - Implement SHAP or LIME to understand the importance of each feature in model predictions.

- **Clinical Applications:** 
  - Focus on ensuring that model decisions are transparent, especially for medical use cases.

---

### Phase 7: Long-Term Monitoring and Continuous Learning

- **Model Updates:** 
  - After deployment, periodically collect new data and update the model to adapt to changing gait patterns.

- **Incremental Learning:** 
  - Consider online learning techniques to enable the model to adapt without retraining from scratch.

---

## Installation

Ensure you have the necessary dependencies installed:

```bash
pip install -r requirements.txt
