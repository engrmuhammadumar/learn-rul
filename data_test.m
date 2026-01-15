% Load the data
load('GF1320_Vibration_100mV1.mat');

% Extract sampling frequency
Fs = 25600;  % Fs = 25600 Hz

% Extract one channel (e.g., first channel)
signal_channel = signal(1, :);  % Extract first row (channel 1)

% Create time axis
t = (0:length(signal_channel)-1) / Fs;  % Time vector in seconds

% Plot the signal
figure;
plot(t, signal_channel,'Color', [1 0.5 0], 'LineWidth', 2); % Blue color for signal
xlabel('Time (seconds)', 'FontSize', 22, 'FontWeight', 'bold');
ylabel('Amplitude (g)', 'FontSize', 22, 'FontWeight', 'bold');
%title('Vibration Signal - Channel 1');
% Improve grid and overall appearance
%grid on;
set(gca, 'FontSize', 16, 'FontWeight', 'bold');  % Makes axis numbers bold and readable
%grid on;
