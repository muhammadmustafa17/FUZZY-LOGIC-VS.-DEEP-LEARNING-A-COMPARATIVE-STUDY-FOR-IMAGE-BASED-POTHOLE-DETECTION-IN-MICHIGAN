function pothole_fuzzy_heatmap(imgName)
tic;  
if nargin < 1
    imgName = 'pothole897.jpg';
    %imgName = 'normal6.jpg';
end

clearvars -except imgName; clc; close all;

    %% 1. Load image & preprocessing
    img = imread(imgName);
    fprintf('Loaded image "%s"\n', imgName);

    img_gray = rgb2gray(img);
    img_gray = imgaussfilt(img_gray, 1);
    img_gray = adapthisteq(img_gray, 'ClipLimit', 0.02, 'Distribution', 'rayleigh');

    [h, w] = size(img_gray);
    fprintf('Image size = %dx%d\n', h, w);

    %% 2. Sliding window & feature extraction

    baseWin  = 64;
    win      = min(baseWin, floor(min(h,w)/2));
    step     = max(8, floor(win/2));

    fprintf('Using win = %d, step = %d\n', win, step);

    Nrows = floor((h - win) / step) + 1;
    Ncols = floor((w - win) / step) + 1;
    N     = max(0, Nrows * Ncols);

    if N == 0
        error('Window too large for this image. Reduce win/step.');
    end

    fprintf('Number of patches = %d (%d x %d)\n', N, Nrows, Ncols);

    I_vals   = zeros(N,1);
    sig_vals = zeros(N,1);
    E_vals   = zeros(N,1);
    A_vals   = zeros(N,1);
    centers  = zeros(N,2);
    idx      = 1;

    for y = 1:step:(h-win+1)
        for x = 1:step:(w-win+1)
            patch = img_gray(y:y+win-1, x:x+win-1);

            I_vals(idx)   = mean(patch(:));
            sig_vals(idx) = std(double(patch(:)));

            edges         = edge(patch,'canny');
            E_vals(idx)   = sum(edges(:))/numel(edges);

            BW = imbinarize(patch,'adaptive',...
                            'ForegroundPolarity','dark',...
                            'Sensitivity',0.60);
            stats = regionprops(BW,'Area','MajorAxisLength','MinorAxisLength');

            if ~isempty(stats)
                [~,bigIdx] = max([stats.Area]);
                if stats(bigIdx).Area > 0.02*numel(patch)
                    A_vals(idx) = stats(bigIdx).MinorAxisLength / stats(bigIdx).MajorAxisLength;
                else
                    A_vals(idx) = 1.0;
                end
            else
                A_vals(idx) = 1.0;
            end

            centers(idx,:) = [y + win/2, x + win/2];
            idx = idx + 1;
        end
    end

    I_vals(idx:end)    = [];
    sig_vals(idx:end)  = [];
    E_vals(idx:end)    = [];
    A_vals(idx:end)    = [];
    centers(idx:end,:) = [];

    %% 3. Normalize features to [0,1] for the FIS

    I_n = (I_vals   - min(I_vals))   ./ (max(I_vals)   - min(I_vals)   + eps);
    S_n = (sig_vals - min(sig_vals)) ./ (max(sig_vals) - min(sig_vals) + eps);
    E_n = (E_vals   - min(E_vals))   ./ (max(E_vals)   - min(E_vals)   + eps);
    A_n = max(0, min(1, (A_vals - 0.4) / 0.6));   % circularity

    %% 4. Build / load Mamdani FIS and evaluate

    fis = readfis('PotholeFIS_final.fis');   

    scores = zeros(length(I_n),1);
    for i = 1:length(I_n)
        scores(i) = evalfis(fis,[I_n(i) S_n(i) E_n(i) A_n(i)]);
    end

    fprintf('FIS scores: min = %.3f, max = %.3f, mean = %.3f\n', ...
        min(scores), max(scores), mean(scores));

    %% 5. Image-level decision

    maxScore  = max(scores);
    meanScore = mean(scores);
    contrast  = maxScore - meanScore;

    img_thresh      = 0.5;
    contrast_thresh = 0.10;

    if (maxScore >= img_thresh) && (contrast >= contrast_thresh)
        img_decision = 'DETECTED';
    else
        img_decision = 'NOT DETECTED';
    end

    fprintf('IMAGE-LEVEL DECISION: %s (max = %.3f, contrast = %.3f)\n', ...
        img_decision, maxScore, contrast);

    %% 6. Confidence map & overlay

    heatmap = zeros(h,w);
    for i = 1:length(scores)
        cy = round(centers(i,1));
        cx = round(centers(i,2));

        y1 = max(1, cy - win/2);
        y2 = min(h, cy + win/2 - 1);
        x1 = max(1, cx - win/2);
        x2 = min(w, cx + win/2 - 1);

        heatmap(y1:y2,x1:x2) = max(heatmap(y1:y2,x1:x2), scores(i));
    end

    heat_norm = heatmap;

    if strcmp(img_decision,'DETECTED')
        T = graythresh(heat_norm);
        binary = heat_norm >= T;

        se = strel('disk', max(1, round(win/16)));
        binary = imclose(binary,se);
        binary = imopen(binary,se);
    else
        binary = false(h,w);
    end

    result = img;
    mask   = repmat(binary,[1 1 3]);
    result(mask) = uint8(0.4*double(result(mask)) + 0.6*255);

    %% 7. Display
    figure('Position',[100 100 1500 500]);

    subplot(1,3,1);
    imshow(img); title('Original Image');

    subplot(1,3,2);
    imagesc(heat_norm); axis image off;
    colormap(gca,'jet'); colorbar;
    title('Mamdani FIS Confidence Map');

    subplot(1,3,3);
    imshow(result);
    title(['Detected Pothole Regions (', img_decision, ')']);
    elapsed_time = toc;
    disp(['Code execution time: ', num2str(elapsed_time), ' seconds']);
end
