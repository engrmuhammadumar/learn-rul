%% CWRU Bearing Dataset - CWT Image Generation (Handles Nested Cells)
% Works with nested cell arrays in CW fields
% Author: Muhammad Umar
% Date: 2025

clear all; close all; clc;

%% Configuration
INPUT_PATH = 'F:\NeuTech\CWRU';
OUTPUT_BASE_PATH = 'F:\NeuTech\Signals\with axis\cwru';
NUM_SAMPLES_PER_FILE = 2;  % Number of random segments per file

% CWT Parameters
WAVELET = 'amor';
SAMPLING_FREQUENCY = 12000;  % Hz
SEGMENT_LENGTH = 10000;  % Samples per segment

%% Main Processing
fprintf('==================================================\n');
fprintf('CWRU Bearing Dataset - CWT Image Generation\n');
fprintf('==================================================\n');
fprintf('Input path: %s\n', INPUT_PATH);
fprintf('Output path: %s\n', OUTPUT_BASE_PATH);
fprintf('Wavelet: %s\n', WAVELET);
fprintf('Sampling frequency: %d Hz\n', SAMPLING_FREQUENCY);
fprintf('Samples per file: %d\n', NUM_SAMPLES_PER_FILE);
fprintf('==================================================\n\n');

% Create base output directory
if ~exist(OUTPUT_BASE_PATH, 'dir')
    mkdir(OUTPUT_BASE_PATH);
end

% Define file and field mappings
file_map = struct();
file_map.healthy.file = 'healthy.mat';
file_map.healthy.field = 'Healthy_CW';

file_map.ball.file = 'ball.mat';
file_map.ball.field = 'Ball_CW';

file_map.inner.file = 'inner.mat';
file_map.inner.field = 'Inner_CW';

file_map.outer.file = 'outer.mat';
file_map.outer.field = 'Outer_CW';

classes = fieldnames(file_map);

% Process each class
for c = 1:length(classes)
    class_name = classes{c};
    file_name = file_map.(class_name).file;
    field_name = file_map.(class_name).field;
    file_path = fullfile(INPUT_PATH, file_name);
    
    fprintf('\n========================================\n');
    fprintf('Processing class: %s\n', class_name);
    fprintf('File: %s\n', file_name);
    fprintf('Field: %s\n', field_name);
    fprintf('========================================\n');
    
    if ~exist(file_path, 'file')
        fprintf('❌ File not found: %s\n', file_path);
        continue;
    end
    
    try
        % Load and process file
        process_file(class_name, file_path, field_name, OUTPUT_BASE_PATH, ...
                    NUM_SAMPLES_PER_FILE, WAVELET, SAMPLING_FREQUENCY, SEGMENT_LENGTH);
    catch ME
        fprintf('❌ Error processing %s: %s\n', file_name, ME.message);
        fprintf('   Location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    end
end

fprintf('\n==================================================\n');
fprintf('✅ Processing complete!\n');
fprintf('==================================================\n');

%% Function: Process File
function process_file(class_name, file_path, field_name, output_base, num_samples, ...
                     wavelet, fs, segment_len)
    
    % Create output directory
    output_path = fullfile(output_base, class_name);
    if ~exist(output_path, 'dir')
        mkdir(output_path);
    end
    
    % Load all signals
    fprintf('Loading signals from field: %s...\n', field_name);
    all_signals = load_all_signals(file_path, field_name);
    
    if isempty(all_signals)
        fprintf('❌ Failed to load any signals\n');
        return;
    end
    
    fprintf('✓ Loaded %d signals\n', length(all_signals));
    
    % Process each signal
    for sig_idx = 1:min(num_samples, length(all_signals))
        signal_data = all_signals{sig_idx};
        
        fprintf('\n--- Signal %d/%d ---\n', sig_idx, min(num_samples, length(all_signals)));
        fprintf('Signal length: %d samples\n', length(signal_data));
        fprintf('Min: %.4f, Max: %.4f, Mean: %.4f, Std: %.4f\n', ...
                min(signal_data), max(signal_data), mean(signal_data), std(signal_data));
        
        % Check if signal is long enough
        actual_segment_len = min(segment_len, length(signal_data));
        if actual_segment_len < 1000
            fprintf('⚠️  Signal too short (%d samples), skipping\n', actual_segment_len);
            continue;
        end
        
        % Extract segment from beginning
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
        output_filename = sprintf('%s_signal_%d_cwt.png', class_name, sig_idx);
        output_filepath = fullfile(output_path, output_filename);
        
        % Plot and save
        fprintf('Generating plot...\n');
        plot_cwt_image(cfs, f, signal_normalized, fs, output_filepath);
        
        fprintf('✅ Saved: %s\n', output_filename);
    end
end

%% Function: Load All Signals (Handles Nested Cells)
function all_signals = load_all_signals(file_path, field_name)
    all_signals = {};
    
    try
        % Load .mat file
        fprintf('  Loading .mat file...\n');
        data = load(file_path);
        
        % Check if field exists
        if ~isfield(data, field_name)
            fprintf('  ❌ Field %s not found!\n', field_name);
            return;
        end
        
        % Get the field data
        field_data = data.(field_name);
        fprintf('  Field type: %s, Size: %s\n', class(field_data), mat2str(size(field_data)));
        
        % Recursively extract all numeric arrays
        all_signals = extract_all_numeric(field_data, 1);
        
        fprintf('  ✓ Found %d numeric signals\n', length(all_signals));
        
    catch ME
        fprintf('  ❌ Error: %s\n', ME.message);
    end
end

%% Function: Recursively Extract All Numeric Arrays
function signals = extract_all_numeric(data, depth)
    signals = {};
    
    % Prevent infinite recursion
    if depth > 10
        return;
    end
    
    if isnumeric(data)
        % Found numeric data
        if numel(data) > 100  % At least 100 samples
            signals{end+1} = data(:);
        end
    elseif iscell(data)
        % Recursively search cell array
        for i = 1:numel(data)
            item = data{i};
            sub_signals = extract_all_numeric(item, depth + 1);
            signals = [signals, sub_signals];
        end
    elseif isstruct(data)
        % Search struct fields
        fields = fieldnames(data);
        for f = 1:length(fields)
            item = data.(fields{f});
            sub_signals = extract_all_numeric(item, depth + 1);
            signals = [signals, sub_signals];
        end
    end
end

%% Function: Normalize Signal
function signal_normalized = normalize_signal(signal_data)
    % Remove DC component
    signal_data = signal_data - mean(signal_data);
    
    % Normalize to [-1, 1]
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