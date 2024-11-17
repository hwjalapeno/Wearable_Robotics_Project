clear all;
close all;

%% Parameters Initialization
num_subjects = 11; % Number of subjects per condition (Healthy + Abnormal)
num_channels = 4; % Four channels for four muscles: RF, BF, VM, ST
muscle_names = {'BF', 'RF', 'ST', 'VM'};
conditions = {'Healthy', 'Abnormal'}; % Conditions representing Healthy and Abnormal gait
fs = 1000; % Sampling frequency in Hz
hp_cutoff = 20; % High-pass filter cutoff frequency (20 Hz)
lp_cutoff = 450; % Low-pass filter cutoff frequency (450 Hz)
notch_freq = 50; % Notch filter frequency (50 Hz)
window_size = 50; % Window size in samples for smoothing

% Directory structure
main_dir = '/Users/varunchandrashekarraghavendra/Desktop/NEU/Wearable Robotics/Final Project/Modified Dataset'; % Update with actual path

% Initialize storage for all features and labels
features_Healthy = [];
features_Abnormal = [];
labels_Healthy = [];
labels_Abnormal = [];

function [TD_features, FD_features, wavelet_features] = extract_features(emg_envelope, fs)
    % Time-Domain Features (TD)
    mean_val = mean(emg_envelope);
    std_val = std(emg_envelope);
    iav = sum(abs(emg_envelope));
    emav = sum(abs(emg_envelope)) / length(emg_envelope);
    mav = mean(abs(emg_envelope));
    wl = sum(abs(diff(emg_envelope)));
    aac = mean(abs(diff(emg_envelope)));
    rms_val = rms(emg_envelope);
    mma_val = (sum(abs(emg_envelope).^0.5) / length(emg_envelope)).^2;
    ssi = sum(emg_envelope.^2);
    log_detector = exp(mean(log(abs(emg_envelope) + eps))); % To prevent log(0)

    % Frequency-Domain Features (FD)
    psd = abs(fft(emg_envelope)).^2;
    mean_freq = meanfreq(psd, fs);
    median_freq = medfreq(psd, fs);
    [~, peak_idx] = max(psd);
    peak_freq = (peak_idx - 1) * (fs / length(psd));
    total_power = sum(psd);
    spectral_entropy = wentropy(psd, 'shannon');
    peak_to_rms_ratio = max(abs(emg_envelope)) / rms_val;

    % Wavelet Features (WF) - Using a sample wavelet transform (e.g., db4)
    [c, l] = wavedec(emg_envelope, 4, 'db4');  % Decompose signal into 4 levels
    wavelet_energy = sum(c.^2);  % Wavelet energy (sum of coefficients squared)
    wavelet_entropy = wentropy(c, 'shannon');  % Wavelet entropy
    wavelet_max = max(c);  % Maximum wavelet coefficient

    % Combine all features
    TD_features = [mean_val, std_val, iav, emav, mav, wl, aac, rms_val, ...
                mma_val, ssi, log_detector];
    FD_features = [mean_freq, median_freq, peak_freq, total_power, spectral_entropy, peak_to_rms_ratio];
    wavelet_features = [wavelet_energy, wavelet_entropy, wavelet_max];
end

%% Load and Process Data
for cond_idx = 1:length(conditions)
    condition = conditions{cond_idx};
    
    for subj = 1:num_subjects
        subject_features = [];
        
        for muscle_idx = 1:num_channels
            muscle = muscle_names{muscle_idx};
            data_dir = fullfile(main_dir, condition, muscle);
            file_list = dir(fullfile(data_dir, '*.wav'));
            
            if length(file_list) < num_subjects
                warning('Not enough files for %s condition and muscle %s for subject %d.', condition, muscle, subj);
                continue;
            end
            
            file_name = fullfile(data_dir, file_list(subj).name);
            fprintf('Processing file: %s\n', file_name);
            [data, file_fs] = audioread(file_name);
            
            if file_fs ~= fs
                data = resample(data, fs, file_fs);
            end
            
            emg_signal = data(:, 1)';
            
            %% Normalize and Filter the Signal
            emg_normalized = emg_signal / max(abs(emg_signal)); % MVC normalization
            emg_filtered = preprocess_emg(emg_normalized, fs, hp_cutoff, lp_cutoff, notch_freq);
            
            %% Rectification and Smoothing for Envelope
            emg_rectified = abs(emg_filtered);
            emg_envelope = movmean(emg_rectified, window_size);
            
            %% Feature Extraction
            [TD_features, FD_features, wavelet_features] = extract_features(emg_envelope, fs);  % Pass fs here
            
            % Ensure consistent feature sizes across all channels
            if isempty(subject_features)
                % Initialize the feature matrix with the first set of features
                subject_features = [TD_features, FD_features, wavelet_features];
            else
                % Check that the current features match the size of previous ones
                if size(subject_features, 2) == size([TD_features, FD_features, wavelet_features], 2)
                    subject_features = [subject_features; [TD_features, FD_features, wavelet_features]];
                else
                    warning('Feature dimensions mismatch. Skipping this muscle data.');
                    continue; % Skip this subject-muscle pair if dimensions don't match
                end
            end
        end
        
        % Only add to features if subject features were successfully extracted
        if ~isempty(subject_features)
            if strcmp(condition, 'Healthy')
                features_Healthy = [features_Healthy; subject_features];
                labels_Healthy = [labels_Healthy; 0];
            else
                features_Abnormal = [features_Abnormal; subject_features];
                labels_Abnormal = [labels_Abnormal; 1];
            end
        end
    end
end

%% Concatenate Healthy and Abnormal Features
features = [features_Healthy; features_Abnormal];
labels = [labels_Healthy; labels_Abnormal];

%% Set Training-Test Ratio
train_test_ratio = 0.727;
cv_part = cvpartition(labels, 'HoldOut', 1 - train_test_ratio);
X_train = features(training(cv_part), :);
y_train = labels(training(cv_part));
X_test = features(test(cv_part), :);
y_test = labels(test(cv_part), :);

% Define muscle names and classifier names
muscle_names = {'BF', 'VM', 'ST', 'RF'};
classifier_names = {'KNN', 'SVM', 'Random Forest', 'Decision Tree'};  % Removed Naive Bayes

%% Fixing the indices based on total available features
TD_indices = 1:11;  % Time-domain features (11 features)
FD_indices = 12:17;  % Frequency-domain features (6 features)
WF_indices = 18:20;  % Wavelet features (3 features, corrected to 18:20)

% Initialize the results table with one row for each muscle and columns for each domain accuracy
num_muscles = length(muscle_names);
num_classifiers = length(classifier_names);

% Number of accuracy columns per classifier (TD, FD, WF)
accuracy_columns_per_classifier = 3;

% Generate the accuracy column names dynamically
accuracy_column_names = {};
for i = 1:num_classifiers
    classifier_name = classifier_names{i};
    accuracy_column_names = [accuracy_column_names, ...
                              strcat(classifier_name, '_TD_Accuracy'), ...
                              strcat(classifier_name, '_FD_Accuracy'), ...
                              strcat(classifier_name, '_WF_Accuracy')];
end

% Initialize the results table with the correct number of columns
results_table = table('Size', [num_muscles, 1 + length(accuracy_column_names)], ...
                      'VariableTypes', [{'string'}, repmat({'double'}, 1, length(accuracy_column_names))], ...
                      'VariableNames', ['Muscle', accuracy_column_names]);

% Loop through each muscle to populate the results
for j = 1:num_muscles
    muscle_name = muscle_names{j};
    
    % Initialize a row for the current muscle
    results_row = {muscle_name};
    
    % Loop through each classifier
    for i = 1:num_classifiers
        classifier_name = classifier_names{i};
        
        % Choose a model for each classifier
        switch classifier_name
            case 'KNN'
                TD_model = fitcknn(X_train(:, TD_indices), y_train);
                FD_model = fitcknn(X_train(:, FD_indices), y_train);
                WF_model = fitcknn(X_train(:, WF_indices), y_train);
            case 'SVM'
                TD_model = fitcsvm(X_train(:, TD_indices), y_train);
                FD_model = fitcsvm(X_train(:, FD_indices), y_train);
                WF_model = fitcsvm(X_train(:, WF_indices), y_train);
            case 'Random Forest'
                TD_model = fitcensemble(X_train(:, TD_indices), y_train, 'Method', 'Bag');
                FD_model = fitcensemble(X_train(:, FD_indices), y_train, 'Method', 'Bag');
                WF_model = fitcensemble(X_train(:, WF_indices), y_train, 'Method', 'Bag');
            case 'Decision Tree'
                TD_model = fitctree(X_train(:, TD_indices), y_train);
                FD_model = fitctree(X_train(:, FD_indices), y_train);
                WF_model = fitctree(X_train(:, WF_indices), y_train);
        end
        
        % Evaluate each model
        TD_predictions = predict(TD_model, X_test(:, TD_indices));
        FD_predictions = predict(FD_model, X_test(:, FD_indices));
        WF_predictions = predict(WF_model, X_test(:, WF_indices));
        
        % Calculate accuracies for this muscle and classifier
        TD_accuracy = sum(TD_predictions == y_test) / length(y_test) * 100;
        FD_accuracy = sum(FD_predictions == y_test) / length(y_test) * 100;
        WF_accuracy = sum(WF_predictions == y_test) / length(y_test) * 100;
        
        % Add the accuracies to the results row for the current muscle
        results_row = [results_row, TD_accuracy, FD_accuracy, WF_accuracy];
    end
    
    % Add the current muscle's result row to the results table
    results_table(j, :) = results_row;
end

% Write the results to a CSV file
output_file = 'muscle_classification_results.csv';
writetable(results_table, output_file);
fprintf('Results have been saved to %s\n', output_file);

% Helper function to remove zero-variance columns
function [X_filtered, zeroVarCols] = removeZeroVariance(X)
    zeroVarCols = var(X) == 0; % Identify zero-variance columns
    X_filtered = X(:, ~zeroVarCols); % Remove zero-variance columns
end

function emg_filtered = preprocess_emg(emg_signal, fs, hp_cutoff, lp_cutoff, notch_freq)
    % High-pass filter
    [b_hp, a_hp] = butter(4, hp_cutoff / (fs / 2), 'high');
    % Low-pass filter
    [b_lp, a_lp] = butter(4, lp_cutoff / (fs / 2), 'low');
    % Notch filter: The second parameter (Q-factor) is set to 30, not the bandwidth
    Q = 30; % Quality factor for notch filter
    Wo = notch_freq / (fs / 2); % Normalized notch frequency
    BW = Wo / Q; % Bandwidth for the notch filter

    % Apply notch filter
    [b_notch, a_notch] = iirnotch(Wo, BW); 

    % Apply filters in sequence
    emg_filtered = filter(b_notch, a_notch, emg_signal);
    emg_filtered = filter(b_hp, a_hp, emg_filtered);
    emg_filtered = filter(b_lp, a_lp, emg_filtered);
end

%% Perform k-Fold Cross-Validation
k = 10; % Number of folds
cv = cvpartition(labels, 'KFold', k);

% Initialize variables to store accuracies for each fold
overall_results = [];

% Loop through each fold
for fold = 1:k
    fprintf('Processing fold %d/%d...\n', fold, k);

    % Get training and testing indices for the current fold
    train_idx = training(cv, fold);
    test_idx = test(cv, fold);

    % Create training and testing sets for this fold
    X_train = features(train_idx, :);
    y_train = labels(train_idx);
    X_test = features(test_idx, :);
    y_test = labels(test_idx);

    % Loop through each muscle
    for j = 1:num_muscles
        muscle_name = muscle_names{j};

        % Loop through each classifier
        for i = 1:num_classifiers
            classifier_name = classifier_names{i};

            % Choose a model for each classifier
            switch classifier_name
                case 'KNN'
                    TD_model = fitcknn(X_train(:, TD_indices), y_train);
                    FD_model = fitcknn(X_train(:, FD_indices), y_train);
                    WF_model = fitcknn(X_train(:, WF_indices), y_train);
                case 'SVM'
                    TD_model = fitcsvm(X_train(:, TD_indices), y_train);
                    FD_model = fitcsvm(X_train(:, FD_indices), y_train);
                    WF_model = fitcsvm(X_train(:, WF_indices), y_train);
                case 'Random Forest'
                    TD_model = fitcensemble(X_train(:, TD_indices), y_train, 'Method', 'Bag');
                    FD_model = fitcensemble(X_train(:, FD_indices), y_train, 'Method', 'Bag');
                    WF_model = fitcensemble(X_train(:, WF_indices), y_train, 'Method', 'Bag');
                case 'Decision Tree'
                    TD_model = fitctree(X_train(:, TD_indices), y_train);
                    FD_model = fitctree(X_train(:, FD_indices), y_train);
                    WF_model = fitctree(X_train(:, WF_indices), y_train);
            end

            % Evaluate each model
            TD_predictions = predict(TD_model, X_test(:, TD_indices));
            FD_predictions = predict(FD_model, X_test(:, FD_indices));
            WF_predictions = predict(WF_model, X_test(:, WF_indices));

            % Calculate accuracies
            TD_accuracy = sum(TD_predictions == y_test) / length(y_test) * 100;
            FD_accuracy = sum(FD_predictions == y_test) / length(y_test) * 100;
            WF_accuracy = sum(WF_predictions == y_test) / length(y_test) * 100;

            % Store results
            overall_results = [overall_results; {muscle_name, classifier_name, fold, ...
                              TD_accuracy, FD_accuracy, WF_accuracy}];
        end
    end
end

% Convert results to a table for easier analysis
results_table = cell2table(overall_results, ...
    'VariableNames', {'Muscle', 'Classifier', 'Fold', ...
                      'TD_Accuracy', 'FD_Accuracy', 'WF_Accuracy'});

% Calculate average accuracy per classifier and muscle across folds
avg_results = varfun(@mean, results_table, ...
    'InputVariables', {'TD_Accuracy', 'FD_Accuracy', 'WF_Accuracy'}, ...
    'GroupingVariables', {'Muscle', 'Classifier'});

% Save results to CSV
output_file = 'kfold_cross_validation_results.csv';
writetable(avg_results, output_file);
fprintf('k-fold cross-validation results saved to %s\n', output_file);
