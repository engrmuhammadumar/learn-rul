function analyze_wfs_all_in_one()
% ANALYZE_WFS_ALL_IN_ONE — Streaming edition (handles 100+ GB files safely)
% - Probes file type by magic bytes
% - For unknown binaries: detects header offset, dtype, endian, channels using small windows
% - Streams file in chunks to compute stats (no full-file load)
% - Captures a small slice for plots; optional CSV is skipped for massive files
%
% Usage:
%   - Set baseFolder to your directory
%   - Run: analyze_wfs_all_in_one

%% ===== USER SETTINGS =====
baseFolder = 'D:\Pipeline RUL Data';     % <<-- change if needed
fileList   = {'B.wfs','C.wfs','D.wfs','E.wfs'};
exportCSV  = false;                       % streaming CSV disabled by default for huge files
plotSliceSamples = 200000;                % samples per channel to plot (first slice)
chunkTargetBytes = 64e6;                  % ~64 MB streaming chunk target
%% =========================

timestamp = datestr(now,'yyyy-mm-dd_HHMMSS');
outRoot   = fullfile(baseFolder, ['wfs_report_' timestamp]);
if ~exist(outRoot,'dir'), mkdir(outRoot); end

summaryRows = {};

fprintf('\n=== WFS Analyzer (Streaming): %s ===\nOutput: %s\n', datestr(now), outRoot);

for k = 1:numel(fileList)
    fname = fileList{k};
    fpath = fullfile(baseFolder, fname);
    fprintf('\n-----------------------------\nProcessing: %s\n', fpath);
    S = struct(); S.file = fname; S.fullpath = fpath;
    if ~isfile(fpath)
        warning('File missing: %s', fpath);
        S.status = 'missing'; summaryRows{end+1} = S; %#ok<AGROW>
        continue
    end

    dinfo = dir(fpath); S.bytes = dinfo.bytes;

    b = read_head(fpath, 512);
    [tg, txtish] = guess_type(b);
    S.type_guess  = char(tg);
    S.ascii_ratio = txtish;
    S.hex64_preview = upper(join_cellstr(arrayfun(@(x) sprintf('%02X',x), b(1:min(64,numel(b))), 'UniformOutput',false),' '));

    outDir = fullfile(outRoot, strrep(fname,'.wfs',''));
    if ~exist(outDir,'dir'), mkdir(outDir); end

    probeTxt = fullfile(outDir, 'probe.txt');
    fid = fopen(probeTxt,'w');
    fprintf(fid,'File: %s\nSize: %d bytes\nTypeGuess: %s\nASCII_ratio: %.3f\nHex64:\n%s\n', ...
        fpath, S.bytes, S.type_guess, S.ascii_ratio, S.hex64_preview);
    fclose(fid);

    try
        switch S.type_guess
            case 'ZIP'
                [S, extras] = handle_zip(fpath, outDir, S);
                plots = {};
            case 'MATv5'
                [S, extras] = handle_mat(fpath, outDir, S);
                plots = {};
            case 'HDF5'
                [S, extras] = handle_hdf5(fpath, outDir, S);
                plots = {};
            case 'WAV'
                [S, extras, plots] = handle_wav(fpath, outDir, S, plotSliceSamples);
            case 'TDMS'
                [S, extras, plots] = handle_tdms(fpath, outDir, S, plotSliceSamples);
            case {'XML','text-ish'}
                [S, extras] = handle_textlike(fpath, outDir, S);
                plots = {};
            otherwise
                % Unknown/binary: streaming detection + streaming stats
                [S, extras, plots] = handle_unknown_streaming(fpath, outDir, S, plotSliceSamples, chunkTargetBytes, exportCSV);
        end
        S.extras = extras; S.status = 'ok'; S.plots = plots;
    catch ME
        S.status = 'error';
        S.error_message = ME.message;
        warning('Error parsing %s: %s', fpath, ME.message);
    end

    savejson(fullfile(outDir, 'summary.json'), S);
    summaryRows{end+1} = S; %#ok<AGROW>
end

write_readme(fullfile(outRoot,'README.md'), summaryRows);
fprintf('\nDone. See: %s\n', outRoot);

end % main


%% ===================== Helpers =====================

function b = read_head(fp, n)
    fid = fopen(fp,'r'); b = fread(fid, n, 'uint8=>uint8')'; fclose(fid);
end

function [tg, asciiRatio] = guess_type(b)
    tg = 'unknown';
    if numel(b)>=4 && isequal(b(1:4), uint8([80 75 3 4]))        % ZIP: PK..
        tg = 'ZIP';
    elseif numel(b)>=8 && isequal(b(1:8), uint8([137 72 68 70 13 10 26 10])) % HDF5
        tg = 'HDF5';
    elseif numel(b)>=12 && isequal(char(b(1:4).'),'RIFF') && isequal(char(b(9:12).'),'WAVE')
        tg = 'WAV';
    elseif numel(b)>=4 && isequal(char(b(1:4).'),'TDSm')
        tg = 'TDMS';
    else
        strHead = char(b(:).'); strHeadTrim = strtrim(strHead);
        if strncmp(strHeadTrim, 'MATLAB 5.0 MAT-file', length('MATLAB 5.0 MAT-file'))
            tg = 'MATv5';
        elseif strncmp(strHeadTrim, '<?xml', length('<?xml'))
            tg = 'XML';
        end
    end
    asciiRatio = mean( (b>=32 & b<=126) | ismember(b,[9 10 13]) );
    if strcmp(tg,'unknown') && asciiRatio > 0.95, tg = 'text-ish'; end
end

function [S, extras] = handle_zip(fp, outDir, S)
    tmpOut = fullfile(outDir, 'unpacked'); if ~exist(tmpOut,'dir'), mkdir(tmpOut); end
    unzip(fp, tmpOut);
    dd = dir(tmpOut); dd = dd(~[dd.isdir]);
    listing = arrayfun(@(d) struct('name',d.name,'bytes',d.bytes,'path',fullfile(d.folder,d.name)), dd);
    S.container = 'zip'; S.container_count = numel(listing);
    metas = {};
    for i=1:numel(listing)
        [~,~,ext] = fileparts(listing(i).name);
        switch lower(ext)
            case '.json'
                try, metas{end+1} = struct('file',listing(i).name,'json',try_jsondecode(fileread(listing(i).path))); end %#ok<AGROW>
            case {'.xml','.xaml'}
                try, metas{end+1} = struct('file',listing(i).name,'xml',try_readstruct(listing(i).path)); end %#ok<AGROW>
            case {'.txt','.cfg','.ini','.log'}
                try, metas{end+1} = struct('file',listing(i).name,'text',read_text_head(listing(i).path, 5000)); end %#ok<AGROW>
        end
    end
    extras = struct('zip_listing',{listing}, 'metadata_preview',{metas});
end

function [S, extras] = handle_mat(fp, outDir, S)
    w = whos('-file', fp);
    S.mat_variables = {w.name}';
    S.mat_sizes     = arrayfun(@(x) x.size, w, 'UniformOutput', false);
    savejson(fullfile(outDir, 'mat_whos.json'), struct('whos',{w}));
    extras = struct();
end

function [S, extras] = handle_hdf5(fp, outDir, S)
    info = h5info(fp); savejson(fullfile(outDir,'h5info.json'), info);
    S.h5_num_groups = numel(info.Groups); S.h5_num_datasets = numel(info.Datasets);
    extras = struct('h5info_saved','h5info.json');
end

function [S, extras, plots] = handle_wav(fp, outDir, S, plotSliceSamples)
    ai = audioinfo(fp); [y, Fs] = audioread(fp, [1, min(ai.TotalSamples, plotSliceSamples)]);
    S.wav_channels = ai.NumChannels; S.wav_fs = ai.SampleRate; S.wav_duration_sec = ai.Duration;
    [plots, stats] = plot_and_stats(y, Fs, outDir, 'wav');
    S.stats = stats; save(fullfile(outDir,'data_head.mat'),'y','Fs','-v7');
    extras = struct('audioinfo', ai);
end

function [S, extras, plots] = handle_tdms(fp, outDir, S, plotSliceSamples)
    plots = {}; extras = struct();
    if exist('tdmsread','file')==2
        T = tdmsread(fp);
        S.tdms_vars = fieldnames(T);
        [y,Fs] = first_numeric_from_struct(T);
        if ~isempty(y)
            y = y(1:min(end,plotSliceSamples),:);
            [plots, stats] = plot_and_stats(y, Fs, outDir, 'tdms');
            S.stats = stats;
        end
        save(fullfile(outDir,'tdms_data_head.mat'),'T','-v7');
    else
        error('TDMS file detected but tdmsread is not available in this MATLAB release.');
    end
end

function [S, extras] = handle_textlike(fp, outDir, S)
    headtxt = read_text_head(fp, 20000);
    S.text_preview = headtxt(1:min(4000, numel(headtxt)));
    extras = struct(); extras.maybe_json = try_jsondecode(headtxt); extras.maybe_xml = try_readstruct(fp);
    copyfile(fp, fullfile(outDir,'raw_text_like.txt'));
end

function [S, extras, plots] = handle_unknown_streaming(fp, outDir, S, plotSliceSamples, chunkTargetBytes, exportCSV)
    % 1) DETECT: try offsets, dtypes, endians, channels using small windows
    offsets  = [0, 512, 1024, 4096, 8192, 16384, 65536];
    dtypes   = {'int16','int32','single','double'};
    dtbytes  = [2,4,4,8];
    endians  = {'ieee-le','ieee-be'};
    channels = 1:4;
    winPerPos = 50000;        % samples per channel per window (for scoring)
    nPositions = 5;           % sample positions across file for scoring

    info = dir(fp); fileBytes = info.bytes;
    bestScore = -inf; bestMeta = []; bestYsample = [];

    for oi = 1:numel(offsets)
        off = offsets(oi);
        if off >= fileBytes - 1024, continue; end
        for di = 1:numel(dtypes)
            for ei = 1:numel(endians)
                for ch = channels
                    yAgg = [];
                    [ok, yAgg] = read_windows(fp, off, dtypes{di}, dtbytes(di), endians{ei}, ch, winPerPos, nPositions);
                    if ~ok, continue; end
                    % scoring
                    s   = std(double(yAgg(:)));
                    z   = mean(yAgg(:)==0);
                    u   = numel(unique(yAgg(1:min(end,1e5))));
                    sat = mean(abs(double(yAgg(:))) >= 0.99*(max(abs(double(yAgg(:))))+eps));
                    score = s - 1e3*z + log(u+1) - 5*sat - 0.1*ch - 0.000001*off; % tiny preference for smaller header
                    if isfinite(score) && score>bestScore && s>0
                        bestScore = score;
                        bestMeta  = struct('offset',off,'dtype',dtypes{di},'dtype_bytes',dtbytes(di), ...
                                           'endian',endians{ei},'channels',ch);
                        bestYsample = yAgg;
                    end
                end
            end
        end
    end

    if isempty(bestMeta)
        error('Could not find a plausible decode (try more offsets/channels).');
    end

    % 2) PLOT SLICE: take first plotSliceSamples frames from after offset
    bytesPerFrame = bestMeta.dtype_bytes*bestMeta.channels;
    nFramesTotal  = floor( (fileBytes - bestMeta.offset) / bytesPerFrame );
    S.decode_offset   = bestMeta.offset;
    S.decode_dtype    = bestMeta.dtype;
    S.decode_endian   = bestMeta.endian;
    S.decode_channels = bestMeta.channels;
    S.decode_nsamples = nFramesTotal;

    yPlot = read_block(fp, bestMeta, 0, min(plotSliceSamples, nFramesTotal)); % NxC
    FsGuess = [];
    try, FsGuess = estimate_fs(yPlot(:,1)); catch, end
    if ~isempty(FsGuess), S.guessed_Fs = FsGuess; end

    % 3) STREAMING STATS across whole file
    [stats, csvNote] = streaming_stats(fp, bestMeta, nFramesTotal, chunkTargetBytes, FsGuess);
    S.stats = stats;
    if ~isempty(csvNote), S.csv_note = csvNote; end

    % 4) PLOTS from yPlot
    plots = plot_and_stats(yPlot, FsGuess, outDir, 'decoded');

    % 5) SAVE small artifacts (not entire file)
    save(fullfile(outDir,'decoded_meta.mat'), 'bestMeta','FsGuess','-v7');
    save(fullfile(outDir,'decoded_head.mat'), 'yPlot','FsGuess','-v7');

    % Optional CSV (only if reasonably small)
    if exportCSV
        maxRowsCSV = 2e7; % safety cap
        if nFramesTotal <= maxRowsCSV
            csvNote = write_csv_streaming(fp, bestMeta, nFramesTotal, outDir, 'decoded_stream.csv', chunkTargetBytes, FsGuess);
            if ~isempty(csvNote), S.csv_note = csvNote; end
        else
            S.csv_note = sprintf('CSV skipped: %,d rows would be too large.', nFramesTotal);
        end
    end

    extras = struct('detection_sample_preview', bestYsample(1:min(end,5*bestMeta.channels),:));
end

function [ok, yAgg] = read_windows(fp, offset, dtype, dtypeBytes, endian, channels, winPerPos, nPositions)
    ok = true; yAgg = [];
    info = dir(fp); F = info.bytes;
    bytesPerFrame = dtypeBytes * channels;
    usable = F - offset;
    if usable <= bytesPerFrame*winPerPos, ok = false; return; end

    % positions across file (relative 0..1)
    rel = linspace(0.05, 0.95, nPositions);
    for i=1:nPositions
        startFrame = floor((usable/bytesPerFrame - winPerPos) * rel(i));
        blk = read_block(fp, struct('offset',offset,'dtype',dtype,'dtype_bytes',dtypeBytes,'endian',endian,'channels',channels), startFrame, winPerPos);
        if isempty(blk) || any(~isfinite(blk(:))), ok = false; return; end
        yAgg = [yAgg; blk]; %#ok<AGROW>
        if numel(yAgg) > 1e6, break; end % keep small
    end
end

function y = read_block(fp, meta, startFrame, nFrames)
    % Reads a contiguous block (nFrames) starting at startFrame (0-based) after offset.
    bytesPerFrame = meta.dtype_bytes * meta.channels;
    startByte = meta.offset + startFrame*bytesPerFrame;

    fid = fopen(fp,'r', meta.endian);
    if fid<0, y = []; return; end
    cleanup = onCleanup(@() fclose(fid));
    fseek(fid, startByte, 'bof');

    % Read as a flat vector then reshape to NxC
    [vec, count] = fread(fid, nFrames*meta.channels, [meta.dtype '=>' meta.dtype]);
    if count < nFrames*meta.channels
        % Truncate to full frames
        nFrames = floor(count / meta.channels);
        vec = vec(1:nFrames*meta.channels);
    end
    if isempty(vec)
        y = [];
    else
        y = reshape(vec, meta.channels, []).';
    end
end

function [stats, note] = streaming_stats(fp, meta, nFramesTotal, chunkTargetBytes, Fs)
    % One pass streaming stats: min/max/mean/std (Welford) per channel.
    note = '';
    bytesPerFrame = meta.dtype_bytes*meta.channels;
    chunkFrames   = max(1, floor(chunkTargetBytes / bytesPerFrame));

    % Welford accumulators
    C = meta.channels;
    n = 0;
    meanC = zeros(1,C);
    M2C   = zeros(1,C);
    minC  =  inf(1,C);
    maxC  = -inf(1,C);

    pos = 0;
    while pos < nFramesTotal
        thisN = min(chunkFrames, nFramesTotal - pos);
        y = read_block(fp, meta, pos, thisN);
        if isempty(y), break; end
        % update min/max
        minC = min(minC, min(y,[],1));
        maxC = max(maxC, max(y,[],1));
        % Welford update
        nOld = n; n = n + size(y,1);
        delta = bsxfun(@minus, double(y), meanC);
        meanC = meanC + sum(delta,1) ./ n;
        delta2 = bsxfun(@minus, double(y), meanC);
        M2C = M2C + sum(delta .* delta2, 1);

        pos = pos + thisN;
    end

    stats = struct();
    stats.samples  = n;
    stats.channels = C;
    stats.min      = minC;
    stats.max      = maxC;
    stats.mean     = meanC;
    stats.std      = sqrt(M2C / max(1,n-1));
    if ~isempty(Fs)
        stats.Fs = Fs;
        stats.duration_sec = n / Fs;
    end
end

function note = write_csv_streaming(fp, meta, nFramesTotal, outDir, filename, chunkTargetBytes, Fs)
    note = '';
    bytesPerFrame = meta.dtype_bytes*meta.channels;
    chunkFrames   = max(1, floor(chunkTargetBytes / bytesPerFrame));
    fn = fullfile(outDir, filename);
    fidw = fopen(fn,'w');
    if fidw<0, note = 'Could not open CSV for writing.'; return; end
    cleanup = onCleanup(@() fclose(fidw));

    % header
    if ~isempty(Fs)
        fprintf(fidw, 'time_s,');
    end
    for c=1:meta.channels
        fprintf(fidw, 'ch%d', c);
        if c<meta.channels, fprintf(fidw, ','); end
    end
    fprintf(fidw, '\n');

    pos = 0;
    while pos < nFramesTotal
        thisN = min(chunkFrames, nFramesTotal - pos);
        y = read_block(fp, meta, pos, thisN);
        if isempty(y), break; end
        if ~isempty(Fs)
            t0 = pos / Fs;
            tt = (0:thisN-1).'/Fs + t0;
            block = [tt, double(y)];
        else
            block = double(y);
        end
        % write block
        fmt = repmat('%.10g,',1, size(block,2)); fmt(end) = '\n';
        for i=1:size(block,1)
            fprintf(fidw, fmt, block(i,:));
        end
        pos = pos + thisN;
    end
end

function [plots, stats] = plot_and_stats(y, Fs, outDir, tag)
    if isvector(y), y = y(:); end
    [N,C] = size(y);
    plots = {};
    stats = struct('samples',N,'channels',C,'min',min(y), 'max',max(y), ...
                   'mean',mean(y), 'std', std(double(y)));
    if ~isempty(Fs), stats.Fs = Fs; stats.duration_sec = N/Fs; end

    % Time-series
    f1 = figure('Visible','off','Name',[tag ' timeseries']); plot(y); grid on
    xlabel('Sample'); ylabel('Amplitude'); title(sprintf('Time Series (%s) - %d samples', tag, N));
    legend(arrayfun(@(i) sprintf('ch%d',i), 1:C, 'UniformOutput', false), 'Location','best');
    p1 = fullfile(outDir, sprintf('%s_timeseries.png', tag)); saveas(f1, p1); close(f1); plots{end+1} = p1;

    % Spectrum (Welch if available, else FFT)
    f2 = figure('Visible','off','Name',[tag ' spectrum']); hold on
    for c=1:C
        try
            if ~isempty(Fs) && exist('pwelch','file')==2
                [pxx,f] = pwelch(double(y(:,c)), [], [], [], Fs);
                plot(f,10*log10(pxx));
                xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)'); title(sprintf('Welch Spectrum (%s)', tag));
            else
                Y  = abs(fft(double(y(:,c))));
                f  = (0:N-1);
                if ~isempty(Fs), f = f*Fs/N; xlabel('Frequency (Hz)'); else, xlabel('FFT bin'); end
                plot(f, 20*log10(Y+eps)); ylabel('Magnitude (dB)'); title(sprintf('FFT Spectrum (%s)', tag));
            end
        catch
        end
    end
    grid on
    p2 = fullfile(outDir, sprintf('%s_spectrum.png', tag)); saveas(f2, p2); close(f2); plots{end+1} = p2;

    % Histogram
    f3 = figure('Visible','off','Name',[tag ' histogram']); hold on
    for c=1:C, histogram(double(y(:,c)), 128); end
    grid on; xlabel('Amplitude'); ylabel('Count'); title(sprintf('Histogram (%s)', tag));
    p3 = fullfile(outDir, sprintf('%s_hist.png', tag)); saveas(f3, p3); close(f3); plots{end+1} = p3;
end

function Fs = estimate_fs(x)
    % Heuristic Fs guess using PSD peaked-ness across candidates
    x = double(x(:)); x = x - mean(x);
    N = min(numel(x), 200000); x = x(1:N);
    Fs_candidates = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000];
    bestScore = -inf; bestFs = [];
    for FsC = Fs_candidates
        [pxx,f] = simple_psd(x, FsC);
        band = (f>=10 & f<=min(10000, FsC/2));
        sc = max(pxx(band)) / (mean(pxx(band))+eps);
        if sc > bestScore, bestScore = sc; bestFs = FsC; end
    end
    if bestScore < 5, Fs = []; else, Fs = bestFs; end
end

function [pxx,f] = simple_psd(x, Fs)
    nfft = 2^nextpow2(min(numel(x), 262144));
    wlen = min(numel(x), nfft);
    xw = x(1:wlen) .* hann(wlen);
    X = fft(xw, nfft); X = X(1:floor(nfft/2)+1);
    pxx = (abs(X).^2) / (sum(hann(wlen).^2) * Fs);
    f = linspace(0, Fs/2, numel(pxx));
end

function [y,Fs] = first_numeric_from_struct(T)
    y = []; Fs = [];
    fn = fieldnames(T);
    for i=1:numel(fn)
        v = T.(fn{i});
        if isnumeric(v), y=v; if isvector(y), y=y(:); end; break
        elseif isstruct(v), [y,Fs]=first_numeric_from_struct(v); if ~isempty(y), break, end
        elseif istable(v), nc=varfun(@isnumeric, v, 'OutputFormat','uniform'); if any(nc), y=table2array(v(:,find(nc,1,'first'))); if isvector(y), y=y(:); end; break, end
        end
    end
end

function txt = read_text_head(fp, N)
    fid = fopen(fp,'r'); c = fread(fid, N, '*char')'; fclose(fid); txt = c;
end

function savejson(path, s)
    try, txt = jsonencode(s); catch, txt = simple_jsonencode(s); end
    fid = fopen(path,'w'); fwrite(fid, txt); fclose(fid);
end

function out = simple_jsonencode(s)
    if isstruct(s)
        if numel(s)>1, parts = arrayfun(@simple_jsonencode, s, 'UniformOutput', false); out = ['[', join_cellstr(parts,','), ']']; return, end
        fn = fieldnames(s); kv = cell(1,numel(fn));
        for i=1:numel(fn), v=s.(fn{i}); kv{i}=['"',escape_json(fn{i}),'":',simple_jsonencode(v)]; end
        out = ['{', join_cellstr(kv,','), '}'];
    elseif ischar(s), out=['"',escape_json(s),'"'];
    elseif isnumeric(s) && isscalar(s), out=num2str(s,17);
    elseif isnumeric(s), out=mat2jsonarray(s);
    elseif iscell(s), parts = cellfun(@simple_jsonencode, s, 'UniformOutput', false); out=['[', join_cellstr(parts,','), ']'];
    elseif islogical(s), out=char("true"*(s~=0)+"false"*(s==0));
    else, out='null';
    end
end

function out = mat2jsonarray(m)
    if isempty(m), out='[]'; return; end
    if isvector(m), parts=arrayfun(@(x) num2str(x,17), m(:).','UniformOutput',false); out=['[',join_cellstr(parts,','),']'];
    else
        rows=cell(size(m,1),1);
        for i=1:size(m,1), parts=arrayfun(@(x) num2str(x,17), m(i,:), 'UniformOutput', false); rows{i}=['[',join_cellstr(parts,','),']']; end
        out=['[',join_cellstr(rows,','),']'];
    end
end

function s = escape_json(x)
    s = strrep(x, '\', '\\'); s = strrep(s, '"', '\"');
    s = strrep(s, sprintf('\n'), '\\n'); s = strrep(s, sprintf('\r'), '\\r'); s = strrep(s, sprintf('\t'), '\\t');
end

function j = join_cellstr(c, sep)
    if isempty(c), j=''; return; end
    c = c(:).'; j=c{1}; for i=2:numel(c), j=[j, sep, c{i}]; end %#ok<AGROW>
end

function out = try_jsondecode(txt)
    try, out = jsondecode(txt); catch, out = []; end
end

function out = try_readstruct(fp)
    out = [];
    try, out = readstruct(fp); catch
        try, xmlread(fp); out = fileread(fp); catch, out = []; end %#ok<NASGU>
    end
end
