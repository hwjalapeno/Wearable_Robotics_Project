# Wearable Robotics Project: Binary Classifier for Gait EMG Signals

This project aims to design a binary classifier for gait Electromyography (EMG) signals using knee muscle data. Below are the steps and suggestions for improving the signal processing, feature extraction, classification, and future work to enhance performance, interpretability, and scalability.

## Project Overview

The goal is to classify EMG signals into two categories based on the knee muscle activity during gait. 

The following approach details the phases of the project:

Data Preprocessing:

Load EMG signal data for each subject, muscle, and condition (Healthy, Abnormal).
Normalize and filter the signals using high-pass, low-pass, and notch filters.
Feature Extraction:

Extract Time-Domain (TD) features: Mean, Standard Deviation, RMS, MAV, WL, etc.
Extract Frequency-Domain (FD) features: Power Spectral Density, Mean Frequency, Median Frequency, Total Power, etc.
Extract Wavelet-Domain (Wavelet) features: Wavelet energy, entropy, and maximum coefficient.
Principal Component Analysis (PCA):

Apply PCA separately on each feature type (TD, FD, Wavelet) to reduce dimensionality and retain 95% variance.
Cross-Validation:

Perform k-fold cross-validation (with k=10) on each feature set (TD, FD, Wavelet) for different classifiers (KNN, SVM, Random Forest, Decision Tree).
Classifier Training:

Train classifiers (KNN, SVM, Random Forest, Decision Tree) using the reduced features.
Evaluate model accuracy using cross-validation.
Results:

Compute and store the mean accuracy for each classifier and feature type.
Save the results (mean accuracy for each classifier and feature set) to a CSV file.


