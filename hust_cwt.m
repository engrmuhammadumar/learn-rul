%% HUST Bearing Dataset - CWT Generation with Proper Classification
% Handles HUST naming convention: B=Ball, IB=Inner, OB=Outer, N=Normal
% Author: Muhammad Umar
% Date: 2025

clear all; close all; clc;

%% Configuration
INPUT_PATH = 'F:\NeuTech\HUST bearing\HUST bearing dataset';
OUTPUT_BASE_PATH = 'F:\NeuTech\Signals\with axis\hust';
NUM_SAMPLES_PER_CLASS = 2;  % Number of images per class

% CWT Parameters
WAVELET = 'amor';
SAMPLING_FREQUENCY = 51200;  % Hz
SEGMENT_LENGTH = 10000;

%% Main Processing
fprintf('==================================================\n');
fprintf('HUST Bearing Dataset - CWT Image Generation\n');
fprintf('==================================================\n');
fprintf('Input path: %s\n', INPUT_PATH);
fprintf('Output path: %s\n', OUTPUT_BASE_PATH);
fprintf('Wavelet: %s\n', WAVELET);
fprintf('Sampling frequency: %d Hz\n', SAMPLING_FREQUENCY);
fprintf('Samples per class: %d\n', NUM_SAMPLES_PER_CLASS);
fprintf('==================================================\n\n');

% Create base output directory
if ~exist(OUTPUT_BASE_PATH, 'dir')
    mkdir(OUTPUT_BASE_PATH);
end

% Get all mat files
fprintf('Searching for .mat files...\n');
mat_files = dir(fullfile(INPUT_PATH, '*.mat'));

if isempty(mat_files)
    fprintf('❌ No .mat files found!\n');
    return;
end

fprintf('✓ Found %d .mat files\n\n', length(mat_files));

% Classify files by HUST naming convention
fprintf('Classifying files by HUST naming convention...\n');
organized = struct();
organized.normal = {};      % N### - Normal/Healthy
organized.ball = {};        % B### - Ball fault
organized.inner = {};       % IB### - Inner race fault
organized.outer = {};       % OB### - Outer race fault
organized.unknown = {};     % Others

for i = 1:length(mat_files)
    file_name = mat_files(i).name;
    file_path = fullfile(INPUT_PATH, file_name);
    
    % Classify based on filename prefix
    if startsWith(upper(file_name), 'N')
        % Normal bearing
        organized.normal{end+1} = struct('name', file_name, 'path', file_path);
    elseif startsWith(upper(file_name), 'IB')
        % Inner race fault
        organized.inner{end+1} = struct('name', file_name, 'path', file_path);
    elseif startsWith(upper(file_name), 'OB')
        % Outer race fault
        organized.outer{end+1} = struct('name', file_name, 'path', file_path);
    elseif startsWith(upper(file_name), 'B') && ~startsWith(upper(file_name), 'BA')
        % Ball fault (but not BA which might be baseline)
        organized.ball{end+1} = struct('name', file_name, 'path', file_path);
    else
        % Unknown pattern
        organized.unknown{end+1} = struct('name', file_name, 'path', file_path);
    end
end

% Display classification results
fprintf('\n==================================================\n');
fprintf('Classification Results:\n');
fprintf('==================================================\n');
fprintf('  Normal/Healthy: %d files\n', length(organized.normal));
fprintf('  Ball Fault:     %d files\n', length(organized.ball));
fprintf('  Inner Race:     %d files\n', length(organized.inner));
fprintf('  Outer Race:     %d files\n', length(organized.outer));
fprintf('  Unknown:        %d files\n', length(organized.unknown));
fprintf('==================================================\n\n');

% Show sample files for each class
class_names = {'normal', 'ball', 'inner', 'outer'};
for c = 1:length(class_names)
    class_name = class_names{c};
    files = organized.(class_name);
    if ~isempty(files)
        fprintf('%s samples: ', upper(class_name));
        for i = 1:min(5, length(files))
            fprintf('%s ', files{i}.name);
        end
        if length(files) > 5
            fprintf('... (%d more)', length(files) - 5);
        end
        fprintf('\n');
    end
end
fprintf('\n');

% Process each class
for c = 1:length(class_names)
    class_name = class_names{c};
    files = organized.(class_name);
    
    if isempty(files)
        fprintf('⚠️  No files for class: %s, skipping\n', class_name);
        continue;
    end
    
    fprintf('\n========================================\n');
    fprintf('Processing class: %s\n', class_name);
    fprintf('Available files: %d\n', length(files));
    fprintf('========================================\n');
    
    try
        process_class(class_name, files, OUTPUT_BASE_PATH, ...
                     NUM_SAMPLES_PER_CLASS, WAVELET, SAMPLING_FREQUENCY, SEGMENT_LENGTH);
    catch ME
        fprintf('❌ Error processing %s: %s\n', class_name, ME.message);
    end
end

fprintf('\n==================================================\n');
fprintf('✅ Processing complete!\n');
fprintf('==================================================\n');
fprintf('Output location: %s\n', OUTPUT_BASE_PATH);

%% Function: Process Class
function process_class(class_name, files, output_base, num_samples, wavelet, fs, segment_len)
    
    % Create output directory
    output_path = fullfile(output_base, class_name);
    if ~exist(output_path, 'dir')
        mkdir(output_path);
    end
    
    % Randomly select files
    rng(42);
    num_to_process = min(num_samples, length(files));
    selected_indices = randperm(length(files), num_to_process);
    
    % Process each selected file
    for idx = 1:num_to_process
        file_idx = selected_indices(idx);
        file_info = files{file_idx};
        
        fprintf('\n--- File %d/%d ---\n', idx, num_to_process);
        fprintf('File: %s\n', file_info.name);
        
        try
            % Load signal
            signal_data = load_hust_signal(file_info.path);
            
            if isempty(signal_data)
                fprintf('⚠️  Failed to load signal, skipping\n');
                continue;
            end
            
            fprintf('Signal length: %d samples\n', length(signal_data));
            fprintf('Min: %.4f, Max: %.4f, Mean: %.4f, Std: %.4f\n', ...
                    min(signal_data), max(signal_data), mean(signal_data), std(signal_data));
            
            % Use segment
            actual_segment_len = min(segment_len, length(signal_data));
            if actual_segment_len < 1000
                fprintf('⚠️  Signal too short, skipping\n');
                continue;
            end
            
            signal_segment = signal_data(1:actual_segment_len);
            
            % Normalize
            signal_normalized = normalize_signal(signal_segment);
            fprintf('Normalized: Min=%.4f, Max=%.4f\n', ...
                    min(signal_normalized), max(signal_normalized));
            
            % Compute CWT
            fprintf('Computing CWT...\n');
            [cfs, f] = cwt(signal_normalized, wavelet, fs);
            
            fprintf('CWT size: %d x %d\n', size(cfs, 1), size(cfs, 2));
            fprintf('Frequency range: %.2f - %.2f Hz\n', min(f), max(f));
            
            % Generate output filename
            [~, base_name, ~] = fileparts(file_info.name);
            output_filename = sprintf('%s_%d_%s_cwt.png', class_name, idx, base_name);
            output_filepath = fullfile(output_path, output_filename);
            
            % Plot and save
            fprintf('Generating plot...\n');
            plot_cwt_image(cfs, f, signal_normalized, fs, output_filepath);
            
            fprintf('✅ Saved: %s\n', output_filename);
            
        catch ME
            fprintf('❌ Error: %s\n', ME.message);
        end
    end
end

%% Function: Load HUST Signal
function signal_data = load_hust_signal(file_path)
    signal_data = [];
    
    try
        data = load(file_path);
        fields = fieldnames(data);
        
        % Priority patterns for HUST dataset
        priority_patterns = {'data', 'vibration', 'signal', 'x', 'y', 'acc', 'DE', 'FE', 'bearing'};
        
        % Try priority patterns
        for p = 1:length(priority_patterns)
            for f = 1:length(fields)
                if contains(lower(fields{f}), lower(priority_patterns{p}))
                    temp = data.(fields{f});
                    if isnumeric(temp) && numel(temp) > 100
                        signal_data = temp(:);
                        return;
                    end
                end
            end
        end
        
        % Try any numeric field
        for f = 1:length(fields)
            if ~startsWith(fields{f}, '__')
                temp = data.(fields{f});
                if isnumeric(temp) && numel(temp) > 100
                    signal_data = temp(:);
                    return;
                end
            end
        end
        
    catch ME
        % Silent fail
    end
end

%% Function: Normalize Signal
function signal_normalized = normalize_signal(signal_data)
    signal_data = signal_data - mean(signal_data);
    max_val = max(abs(signal_data));
    if max_val > 0
        signal_normalized = signal_data / max_val;
    else
        signal_normalized = signal_data;
    end
end

%% Function: Plot CWT Image
function plot_cwt_image(cfs, frequencies, signal_data, fs, output_path)
    % Create figure
    fig = figure('Visible', 'off', 'Position', [100, 100, 800, 600]);
    
    % Time vector
    t = (0:length(signal_data)-1) / fs;
    
    % Plot CWT scalogram
    imagesc(t, frequencies, abs(cfs));
    axis xy;
    
    % Colormap
    colormap('jet');
    
    % Labels with bold font
    xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Frequency (Hz)', 'FontSize', 14, 'FontWeight', 'bold');
    
    % Colorbar
    cb = colorbar;
    ylabel(cb, 'Magnitude', 'FontSize', 12, 'FontWeight', 'bold');
    
    % Make tick labels bold
    ax = gca;
    ax.FontSize = 12;
    ax.FontWeight = 'bold';
    
    % Tight layout
    set(gca, 'Position', [0.1 0.1 0.75 0.85]);
    
    % Save with high resolution
    print(fig, output_path, '-dpng', '-r300');
    
    % Close figure
    close(fig);
end