%% =========================================================
% HUST: Save 2 CWT images per class (PAPER style like your loop)
% Root:   F:\NeuTech\HUST bearing\HUST bearing dataset
% Output: <root>\CWT_2perClass_PAPER\<class>\*.png
%% =========================================================

clear; clc; close all;

ROOT_DIR = "F:\NeuTech\HUST bearing\HUST bearing dataset";
OUT_DIR  = fullfile(ROOT_DIR, "CWT_2perClass_PAPER");
if ~exist(OUT_DIR, "dir"); mkdir(OUT_DIR); end

N_PER_CLASS = 2;

% ===== IMPORTANT: set correct sampling frequency for HUST =====
FS = 51200;  % <-- CHANGE if your HUST Fs is different (Hz)

classes = ["ball","inner","inner+ball","inner+outer","normal","outer","outer+ball"];

% Collect all .mat files
matFiles = dir(fullfile(ROOT_DIR, "**", "*.mat"));
if isempty(matFiles)
    error("No .mat files found under: %s", ROOT_DIR);
end

% Group files by class
filesByClass = containers.Map();
for c = classes
    filesByClass(char(c)) = {};
end

for k = 1:numel(matFiles)
    fpath = fullfile(matFiles(k).folder, matFiles(k).name);
    label = getHUSTLabel(matFiles(k).name);
    if label ~= "unknown"
        tmp = filesByClass(char(label));
        tmp{end+1} = fpath;
        filesByClass(char(label)) = tmp;
    end
end

% Make per-class folders + print counts
fprintf("\nFile counts per class:\n");
for c = classes
    cdir = fullfile(OUT_DIR, char(c));
    if ~exist(cdir, "dir"); mkdir(cdir); end
    fprintf("  %-12s : %d\n", c, numel(filesByClass(char(c))));
end

% ========= MAIN LOOP (like your style) =========
for c = classes
    ckey = char(c);
    flist = filesByClass(ckey);

    if isempty(flist)
        fprintf("\n[WARN] No files for class: %s\n", c);
        continue;
    end

    % Shuffle and take N_PER_CLASS files
    flist = flist(randperm(numel(flist)));
    takeN = min(N_PER_CLASS, numel(flist));

    fprintf("\nSaving %d CWT images for class: %s\n", takeN, c);

    for i = 1:takeN
        fpath = flist{i};

        % ---- Load + normalize signal (same spirit as your loop) ----
        sig = loadFirst1DSignal(fpath);
        sig = sig(:);
        sig = sig - mean(sig);
        sig = sig / (std(sig) + 1e-8);

        % ---- Invisible figure ----
        fig = figure('Visible','off','Color','w','Position',[100 100 950 650]);

        % ---- CWT (amor, Fs) ----
        cwt(sig, 'amor', FS);

        ax = gca;

        % ================================
        % AXIS & FONT STYLING (paper)
        % ================================
        ax.FontSize   = 22;
        ax.FontWeight = 'bold';
        ax.LineWidth  = 2.0;
        ax.TickDir    = 'out';
        ax.Box        = 'on';

        xlabel(ax, 'Time (s)', 'FontSize', 24, 'FontWeight', 'bold');
        ylabel(ax, 'Frequency (Hz)', 'FontSize', 24, 'FontWeight', 'bold');

        title(ax, '');  % remove title only

        % Colorbar styling
        cb = colorbar;
        cb.FontSize   = 20;
        cb.FontWeight = 'bold';
        cb.LineWidth  = 1.8;

        % Keep your blue-ish look
        colormap(turbo);

        axis tight;

        % ---- Save (like your loop) ----
        [~, name, ~] = fileparts(fpath);
        outName = sprintf("%s_%02d_%s.png", ckey, i, name);
        outPath = fullfile(OUT_DIR, ckey, outName);

        saveas(fig, outPath);
        close(fig);

        fprintf("  ✅ %s\n", outPath);
    end
end

fprintf("\nDONE ✅ Saved images in:\n%s\n", OUT_DIR);

%% ====================== helper functions ======================

function label = getHUSTLabel(filename)
    fname = upper(string(filename));
    token = regexp(fname, "^([A-Z]+)\d+", "tokens", "once");
    if isempty(token)
        label = "unknown"; return;
    end

    code = token{1};
    switch code
        case "N"
            label = "normal";
        case "I"
            label = "inner";
        case "O"
            label = "outer";
        case "B"
            label = "ball";
        case "IB"
            label = "inner+ball";
        case "OB"
            label = "outer+ball";
        case "IO"
            label = "inner+outer";
        otherwise
            label = "unknown";
    end
end

function sig = loadFirst1DSignal(fpath)
% Loads a .mat and picks the longest 1D numeric array
    S = load(fpath);
    fn = fieldnames(S);

    bestLen = 0;
    sig = [];

    for j = 1:numel(fn)
        v = S.(fn{j});
        if isnumeric(v)
            x = squeeze(v);
            if isvector(x) && numel(x) > 1000
                if numel(x) > bestLen
                    bestLen = numel(x);
                    sig = double(x);
                end
            end
        end
    end

    if isempty(sig)
        error("No valid 1D numeric signal found in: %s", fpath);
    end
end
