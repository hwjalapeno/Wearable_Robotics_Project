clear all;
close all;

%% Parameters Initialization
% General parameters
num_subjects = 11; % Number of subjects per condition (Healthy + Abnormal)
num_channels = 4; % Four channels for four muscles: RF, BF, VM, ST
muscle_names = {'BF', 'RF', 'ST', 'VM'};
conditions = {'Healthy', 'Abnormal'};
fs = 1000; % Sampling frequency in Hz
hp_cutoff = 20; % High-pass filter cutoff frequency (Hz)
lp_cutoff = 450; % Low-pass filter cutoff frequency (Hz)
notch_freq = 50; % Notch filter frequency (Hz)
window_size = 50; % Window size in samples for smoothing

% Directory structure
main_dir = '/Users/varunchandrashekarraghavendra/Desktop/NEU/Wearable Robotics/Final Project/Modified Dataset';

% Hyperparameters
k_folds = 10; % Number of folds for k-fold cross-validation
knn_neighbors = 5; % Number of neighbors for KNN
svm_kernel = 'linear'; % SVM kernel (changed to 'rbf' for non-linear classification)
svm_box_constraint = 1; % SVM BoxConstraint parameter
svm_gamma = 1; % Gamma for RBF kernel (added as a hyperparameter)
rf_num_trees = 100; % Number of trees for Random Forest
rf_min_leaf_size = 1; % Minimum leaf size for Random Forest
dt_max_splits = 20; % Maximum splits for Decision Tree

% Classifiers
classifiers = {'KNN', 'SVM', 'Random Forest', 'Decision Tree'};

% Initialize storage for all features and labels
features_all = cell(num_channels, 1);
labels_all = cell(num_channels, 1);

%% Load and Process Data
for muscle_idx = 1:num_channels
    muscle = muscle_names{muscle_idx};
    features_muscle = [];
    labels_muscle = [];
    
    for cond_idx = 1:length(conditions)
        condition = conditions{cond_idx};
        label = cond_idx - 1; % Healthy = 0, Abnormal = 1
        
        for subj = 1:num_subjects
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
            
            % Normalize and filter the signal
            emg_normalized = emg_signal / max(abs(emg_signal)); % MVC normalization
            emg_filtered = preprocess_emg(emg_normalized, fs, hp_cutoff, lp_cutoff, notch_freq);
            
            % Rectification and smoothing for envelope
            emg_rectified = abs(emg_filtered);
            emg_envelope = movmean(emg_rectified, window_size);
            
            % Feature extraction
            [TD_features, FD_features, wavelet_features] = extract_features(emg_envelope, fs);
            feature_vector = [TD_features, FD_features, wavelet_features];
            
            % Store features and labels
            features_muscle = [features_muscle; feature_vector];
            labels_muscle = [labels_muscle; label];
        end
    end
    
    features_all{muscle_idx} = features_muscle;
    labels_all{muscle_idx} = labels_muscle;
end

%% k-Fold Cross-Validation for Each Muscle and Classifier
% Initialize results table
results = table('Size', [0, 4], ...
    'VariableTypes', {'string', 'string', 'string', 'double'}, ...
    'VariableNames', {'Muscle', 'Classifier', 'FeatureType', 'MeanAccuracy'});

for muscle_idx = 1:num_channels
    muscle = muscle_names{muscle_idx};
    features = features_all{muscle_idx};
    labels = labels_all{muscle_idx};
    
    % Ensure labels are properly formatted
    if ~isvector(labels)
        labels = labels(:); % Convert to column vector
    end
    if iscell(labels)
        labels = categorical(labels);
    elseif ischar(labels)
        labels = string(labels);
    end
    
    % Split features into domains
    TD_indices = 1:11; % Time-domain features
    FD_indices = 12:17; % Frequency-domain features
    WF_indices = 18:20; % Wavelet features
    
    % Loop through classifiers
    for clf_idx = 1:length(classifiers)
        clf_name = classifiers{clf_idx};
        
        for feature_set = {'TD', 'FD', 'WF'}
            feature_type = feature_set{1};
            switch feature_type
                case 'TD', feature_indices = TD_indices;
                case 'FD', feature_indices = FD_indices;
                case 'WF', feature_indices = WF_indices;
            end
            
            X = features(:, feature_indices);
            cv = cvpartition(labels, 'KFold', k_folds);
            fold_accuracies = zeros(k_folds, 1);
            
            for fold = 1:k_folds
                train_idx = training(cv, fold);
                test_idx = test(cv, fold);
                X_train = X(train_idx, :);
                y_train = labels(train_idx);
                X_test = X(test_idx, :);
                y_test = labels(test_idx);
                
                % Train and predict
                switch clf_name
                    case 'KNN'
                        model = fitcknn(X_train, y_train, 'NumNeighbors', knn_neighbors);
                    case 'SVM'
                        model = fitcsvm(X_train, y_train, 'KernelFunction', svm_kernel, ...
                            'BoxConstraint', svm_box_constraint, 'KernelScale', svm_gamma);
                    case 'Random Forest'
                        % Create a decision tree template with MinLeafSize parameter
                        treeTemplate = templateTree('MinLeafSize', rf_min_leaf_size);
                        % Fit the random forest using the bagging method with the tree template
                        model = fitcensemble(X_train, y_train, 'Method', 'Bag', ...
                            'NumLearningCycles', rf_num_trees, 'Learners', treeTemplate);
                    case 'Decision Tree'
                        model = fitctree(X_train, y_train, 'MaxNumSplits', dt_max_splits);
                end
                
                predictions = predict(model, X_test);
                fold_accuracies(fold) = sum(predictions == y_test) / length(y_test) * 100;
            end
            
            % Calculate mean accuracy across folds
            mean_accuracy = mean(fold_accuracies);
            
            % Only store the mean accuracy after cross-validation
            results = [results; {muscle, clf_name, feature_type, mean_accuracy}];
        end
    end
end

%% Save Results to CSV
output_file = 'mean_accuracy_results.csv';
writetable(results, output_file);
fprintf('Mean accuracy results saved to %s\n', output_file);

%% Helper Functions
function emg_filtered = preprocess_emg(emg_signal, fs, hp_cutoff, lp_cutoff, notch_freq)
    % High-pass filter
    [b_hp, a_hp] = butter(4, hp_cutoff / (fs / 2), 'high');
    % Low-pass filter
    [b_lp, a_lp] = butter(4, lp_cutoff / (fs / 2), 'low');
    % Notch filter
    Q = 30;
    Wo = notch_freq / (fs / 2);
    BW = Wo / Q;
    [b_notch, a_notch] = iirnotch(Wo, BW);

    % Apply filters
    emg_filtered = filter(b_notch, a_notch, emg_signal);
    emg_filtered = filter(b_hp, a_hp, emg_filtered);
    emg_filtered = filter(b_lp, a_lp, emg_filtered);
end

function [TD_features, FD_features, wavelet_features] = extract_features(emg_envelope, fs)
    % Time-Domain Features (TD)
    mean_val = mean(emg_envelope);                % Mean value
    std_val = std(emg_envelope);                  % Standard deviation
    iav = sum(abs(emg_envelope));                 % Integrated absolute value (IAV)
    emav = sum(abs(emg_envelope)) / length(emg_envelope); % Mean absolute value (MAV)
    mav = mean(abs(emg_envelope));                % Mean of absolute values (MAV)
    wl = sum(abs(diff(emg_envelope)));            % Waveform length (WL)
    aac = mean(abs(diff(emg_envelope)));          % Average absolute value of the change (AAC)
    rms_val = rms(emg_envelope);                  % Root mean square (RMS)
    mma_val = (sum(abs(emg_envelope).^0.5) / length(emg_envelope)).^2; % Modified mean absolute (MMA)
    ssi = sum(emg_envelope.^2);                   % Slope sign changes (SSI)
    log_detector = exp(mean(log(abs(emg_envelope) + eps))); % Logarithmic detector (prevent log(0))
    
    % Frequency-Domain Features (FD)
    psd = abs(fft(emg_envelope)).^2;              % Power spectral density (PSD)
    mean_freq = meanfreq(psd, fs);                 % Mean frequency
    median_freq = medfreq(psd, fs);                % Median frequency
    [~, peak_idx] = max(psd);                      % Peak index in PSD
    peak_freq = (peak_idx - 1) * (fs / length(psd)); % Peak frequency
    total_power = sum(psd);                        % Total power
    spectral_entropy = wentropy(psd, 'shannon');   % Spectral entropy
    peak_to_rms_ratio = max(abs(emg_envelope)) / rms_val; % Peak to RMS ratio
    
    % Wavelet Features (WF) - Using a sample wavelet transform (e.g., db4)
    [c, l] = wavedec(emg_envelope, 4, 'db4');    % Decompose signal into 4 levels using db4 wavelet
    wavelet_energy = sum(c.^2);                    % Wavelet energy (sum of squared coefficients)
    wavelet_entropy = wentropy(c, 'shannon');      % Wavelet entropy
    wavelet_max = max(c);                          % Maximum wavelet coefficient
    
    % Combine all features into feature vectors
    TD_features = [mean_val, std_val, iav, emav, mav, wl, aac, rms_val, mma_val, ssi, log_detector];
    FD_features = [mean_freq, median_freq, peak_freq, total_power, spectral_entropy, peak_to_rms_ratio];
    wavelet_features = [wavelet_energy, wavelet_entropy, wavelet_max];
end
