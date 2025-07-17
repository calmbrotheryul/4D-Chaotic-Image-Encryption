function image_encryption_decryption()
    %% Parameter settings
    n = 8; % Block size
    
    %% Read image
    img = imread('Goldhill.tif'); 
    if size(img,3)==3
        img = rgb2gray(img);
    end
    img = double(img);
    original = img;
    
    %% Arnold scrambling
    tic;
    iterations = 50;  % Scrambling iterations (must match decryption)
    scrambled_img = arnold_scramble(img, iterations);
    arnold_time = toc;
    img = scrambled_img; % Update image to scrambled version

    [M, N] = size(img);
    
    %% Check dimensions
    if mod(M,n)~=0 || mod(N,n)~=0
        error('Image dimensions must be divisible by block size');
    end
    
    %% Calculate SHA-256 hash
    hash = get_image_hash(img);
    
    %% Convert hash to initial conditions
    initial_conditions = hash_to_initial(hash, 4);
    
    %% Generate chaotic sequences
    num_blocks = (M/n) * (N/n);
    required_x1 = num_blocks;       % One x1 per block for start row
    required_x2 = num_blocks;       % One x2 per block for start column
    required_x3 = num_blocks * n*n; % n² x3 per block for key stream
    
    % Calculate required time steps
    max_required = max([required_x1, required_x2, required_x3]);
    time_step = 0.01;
    time_vector = 0:time_step:(max_required * time_step);
    
    % Solve chaotic system
    [~, x] = ode45(@your_chaos_system, time_vector, initial_conditions);
    
    % Extract and trim sequences
    x1_seq = x(1:required_x1, 1);
    x2_seq = x(1:required_x2, 2);
    x3_seq = x(1:required_x3, 3);
    
    %% Encryption process
    tic;
    encrypted = encrypt_with_key(img, initial_conditions);
    encrypt_time = toc;
    
    %% Decryption process
    tic;
    decrypted = decrypt_image(encrypted, x1_seq, x2_seq, x3_seq, n, M, N);
    decrypt_time = toc;
    
    %% Inverse Arnold scrambling
    tic;
    original_recovered = arnold_unscramble(decrypted, iterations);
    unarnold_time = toc;
    
    %% Time performance analysis
    fprintf('\n====== Time Performance ======\n');
    fprintf('Arnold scramble time: %.4f s\n', arnold_time);
    fprintf('Encryption time:      %.4f s\n', encrypt_time);
    fprintf('Decryption time:      %.4f s\n', decrypt_time);
    fprintf('Inverse scramble time:%.4f s\n', unarnold_time);
    fprintf('Total time:          %.4f s\n',...
        arnold_time + encrypt_time + decrypt_time + unarnold_time);
     
    %% Display results
    figure;
    subplot(1,3,1), imshow(uint8(original)), title('Original Image');
    subplot(1,3,2), imshow(uint8(encrypted)), title('Encrypted Image');
    subplot(1,3,3), imshow(uint8(original_recovered)), title('Recovered Image');
    
    %% Verify integrity
    psnr_value = psnr(uint8(img), uint8(decrypted));
    fprintf('Decrypted image PSNR: %.2f dB\n', psnr_value);
    
    %% Security analysis
    % 1. Histogram analysis
    plot_histograms(original, encrypted, original_recovered);

    % 2. χ² test
    [chi2_res, chi2_pass] = chi2_test(encrypted);
    fprintf('χ² Test: Statistic=%.2f, Critical=%.2f, Pass=%d\n', ...
        chi2_res.Statistic, chi2_res.Critical, chi2_pass);

    % 3. Correlation analysis
    directions = {'horizontal', 'vertical', 'diagonal'};
    fprintf('Correlation analysis:\n');
    for dir = directions
        orig_corr = correlation_analysis(original, dir{1});
        encr_corr = correlation_analysis(encrypted, dir{1});
        fprintf('  %s: Original=%.4f, Encrypted=%.4f\n', dir{1}, orig_corr, encr_corr);
    end
    plot_correlation_scatter(original, encrypted);
    
    % 4. Information entropy
    entropy_orig = image_entropy(original);
    entropy_encr = image_entropy(encrypted);
    fprintf('Entropy: Original=%.6f, Encrypted=%.6f (Ideal≈7.997)\n', entropy_orig, entropy_encr);

    % 5. Differential attack test
    [npcr, uaci] = differential_attack_test(original);
    fprintf('Differential attack: NPCR=%.4f%%, UACI=%.4f%%\n', npcr, uaci);

    % 6. Key space analysis (theoretical)
    key_space = 256^32 * 1e8 * 1e8; % SHA256 space + chaos precision
    fprintf('Theoretical key space≈2^%.0f\n', log2(key_space));

    %% Attack analysis
    %% Noise attack analysis
    noise_levels = [0.001, 0.01, 0.05, 0.1]; % Noise densities
    results = struct();
    for i = 1:4
        [results(i).noisy_encrypted, results(i).decrypted, results(i).psnr] = ...
            noise_attack_analysis(encrypted, x1_seq, x2_seq, x3_seq, n, M, N, img, noise_levels(i));
    end
    
    % Plot noise attack results
    figure('Name','Noise Attack Analysis', 'Position', [100 100 1600 400]);
    t = tiledlayout(1,4, 'Padding','tight', 'TileSpacing','tight');
    label_pos_y = -0.2;
    for i = 1:4
        nexttile;
        imshow(uint8(results(i).decrypted));
        text(0.5, label_pos_y, ['(', char(96+i), ') '],...
             'Units', 'normalized', 'FontName', 'Times New Roman',...
             'FontSize', 20, 'HorizontalAlignment', 'center');
    end

    %% Cropping attack analysis
    cropped_results = crop_attack_analysis(encrypted, x1_seq, x2_seq, x3_seq, n, M, N, img);
    
    % Plot cropping attack results
    figure('Name','Cropping Attack Analysis', 'Position', [100 100 1200 600]);
    t = tiledlayout(2,4, 'Padding','tight', 'TileSpacing','tight');
    labels = {'(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)'};
    images = {
        cropped_results.rectangle_center.image;
        cropped_results.circle_center.image;
        cropped_results.edge.image;
        cropped_results.band.image;
        cropped_results.rectangle_center.decrypted;
        cropped_results.circle_center.decrypted;
        cropped_results.edge.decrypted;
        cropped_results.band.decrypted
    };
    for i = 1:8
        nexttile;
        imshow(uint8(images{i}));
        text(0.5, -0.15, labels{i}, ...
            'Units', 'normalized', ...
            'FontName', 'Times New Roman', ...
            'FontSize', 20, ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle');
    end

    %% Key sensitivity analysis
    % Part 1: Decryption sensitivity
    figure('Name','Key Sensitivity - Decryption', 'Position',[100 100 1500 800])
    subplot(2,3,1), imshow(uint8(original))
    text(0.5, -0.15, '(a)', 'Units', 'normalized', 'FontName', 'Times New Roman',... 
        'FontSize', 20, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    subplot(2,3,2), imshow(uint8(encrypted))
    text(0.5, -0.15, '(b)', 'Units', 'normalized', 'FontName', 'Times New Roman',... 
        'FontSize', 20, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        
    % Generate 4 different initial conditions
    wrong_initial_conditions = cell(1,4);
    for k = 1:4
        wrong_initial = initial_conditions;
        wrong_initial(k) = wrong_initial(k) + 1e-8;  % Perturb k-th dimension
        wrong_initial_conditions{k} = wrong_initial;
        
        [~, x_wrong] = ode45(@your_chaos_system, time_vector, wrong_initial);
        x1_seq_wrong = x_wrong(1:required_x1,1);
        x2_seq_wrong = x_wrong(1:required_x2,2);
        x3_seq_wrong = x_wrong(1:required_x3,3);
        
        decrypted_wrong = decrypt_image(encrypted, x1_seq_wrong, x2_seq_wrong, x3_seq_wrong, n, M, N);
        decrypted_wrong = arnold_unscramble(decrypted_wrong, iterations);
        
        subplot(2,3,k+2), 
        imshow(uint8(decrypted_wrong)), 
        label = char('c' + k - 1);
        text(0.5, -0.15, ['(' label ')'], 'Units', 'normalized', ...
            'FontName', 'Times New Roman', 'FontSize', 20, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    end

    % Part 2: Encryption sensitivity
    figure('Name','Key Sensitivity - Encryption Differences', 'Position',[100 100 1500 1200])
    subplot(3,3,1), imshow(uint8(encrypted))
    text(0.5, -0.15, '(a)', 'Units', 'normalized', 'FontName', 'Times New Roman',... 
        'FontSize', 20, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        
    for k = 1:4
        initial_modified = wrong_initial_conditions{k};
        [~, x] = ode45(@your_chaos_system, time_vector, initial_modified);
        x1_seq = x(1:required_x1, 1);
        x2_seq = x(1:required_x2, 2);
        x3_seq = x(1:required_x3, 3);
        encrypted_diff = block_processing(scrambled_img, x1_seq, x2_seq, x3_seq, n, M, N, @encrypt_block);
        
        subplot(3,3,k+1), 
        imshow(uint8(encrypted_diff)), 
        label = char('b' + k - 1);
        text(0.5, -0.15, ['(' label ')'], 'Units', 'normalized', ...
            'FontName', 'Times New Roman', 'FontSize', 20, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
            
        subplot(3,3,k+5),
        imshow(imabsdiff(encrypted,encrypted_diff),[]),
        label = char('f' + k - 1);
        text(0.5, -0.15, ['(' label ')'], 'Units', 'normalized', ...
            'FontName', 'Times New Roman', 'FontSize', 20, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
            
        [npcr_diff(k), uaci_diff(k)] = calculate_sensitivity(encrypted, encrypted_diff);
        fprintf('Encryption sensitivity: NPCR=%.4f%%, UACI=%.4f%%\n', npcr_diff(k), uaci_diff(k));
    end
    
    %% Plaintext attack analysis (solid color images)
    pure_black = zeros(512, 'uint8');
    pure_white = 255 * ones(512, 'uint8');
    border_width = 3;
    pure_white(1:border_width, :) = 0;
    pure_white(end-border_width+1:end, :) = 0;
    pure_white(:, 1:border_width) = 0;
    pure_white(:, end-border_width+1:end) = 0;
    
    encrypted_black = encrypt_with_key(double(pure_black), initial_conditions);
    encrypted_white = encrypt_with_key(double(pure_white), initial_conditions);

    figure('Name','Plaintext Attack Analysis','Position',[100 100 1200 800]);
    % ====== Row 1: Pure black image ======
    subplot(2,3,1);
    imshow(pure_black);
    text(0.5, -0.15, '(a)', 'Units', 'normalized', 'FontName', 'Times New Roman',... 
        'FontSize', 20, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

    subplot(2,3,2);
    imshow(uint8(encrypted_black));
    text(0.5, -0.15, '(b)', 'Units', 'normalized', 'FontName', 'Times New Roman',...
        'FontSize', 20, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

    subplot(2,3,3);
    data = uint8(encrypted_black(:));
    [counts, bins] = histcounts(data, 0:256);
    bar(0:255, counts, 'b');
    axis([0 255 0 2500]);
    text(0.5, -0.15, '(c)', ...
        'Units', 'normalized', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 20, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle');
    set(gca, ...
        'FontWeight', 'bold', ...
        'FontSize', 13, ...
        'FontName', 'Times New Roman', ...
        'LineWidth', 1.2 ...
    );

    % ====== Row 2: Pure white image ======
    subplot(2,3,4);
    imshow(pure_white);
    text(0.5, -0.15, '(d)', 'Units', 'normalized', 'FontName', 'Times New Roman',...
        'FontSize', 20, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

    subplot(2,3,5);
    imshow(uint8(encrypted_white));
    text(0.5, -0.15, '(e)', 'Units', 'normalized', 'FontName', 'Times New Roman',...
        'FontSize', 20, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

    subplot(2,3,6);
    data = uint8(encrypted_white(:));
    [counts, bins] = histcounts(data, 0:256);
    bar(0:255, counts, 'b');
    axis([0 255 0 2500]);
    text(0.5, -0.15, '(f)', ...
        'Units', 'normalized', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 20, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle');
    set(gca, ...
        'FontWeight', 'bold', ...
        'FontSize', 13, ...
        'FontName', 'Times New Roman', ...
        'LineWidth', 1.2 ...
    );
end

%% Modified block_processing function (for encryption)
function processed_img = block_processing(img, x1_seq, x2_seq, x3_seq, n, M, N, process_func)
    img = double(img);
    blocks = mat2cell(img, n*ones(1,M/n), n*ones(1,N/n));
    processed_blocks = cell(size(blocks));
    num_blocks = numel(blocks);
    
    state = 0; % Initial state
    
    for i = 1:num_blocks
        block = blocks{i};
        x1_i = x1_seq(i);
        x2_i = x2_seq(i);
        x3_elements = x3_seq( (i-1)*(n*n) +1 : i*(n*n) );
        
        % Process block and get new state
        [processed_block, state] = process_func(block, x1_i, x2_i, x3_elements, n, state);
        processed_blocks{i} = processed_block;
    end
    processed_img = cell2mat(processed_blocks);
end

%% Block encryption function (double diffusion)
function [encrypted_block, new_state] = encrypt_block(block, x1_i, x2_i, x3_elements, n, previous_state)
    % Generate row diffusion key using previous block's state
    key_stream_row = mod(floor( (abs(x3_elements(1:n)) .* x1_i + x2_i + previous_state) * 1e8 ), 256);
    
    % Get start position
    start_row = mod(floor(abs(x1_i)*1e4), n) + 1;
    start_col = mod(floor(abs(x2_i)*1e4), n) + 1;
    seed = sum(floor(abs(x3_elements(1:4))*1e4));
    
    % Generate dynamic zigzag sequence
    zigzag_order = generate_zigzag_sequence(n, [start_row, start_col], seed);
    
    % Scrambling
    scrambled = block_zigzag_shuffle(block, zigzag_order);
    
    % Row diffusion
    for i = 1:n
        scrambled(i, :) = bitxor(scrambled(i, :), key_stream_row(i));
    end
    
    % Column diffusion
    key_stream_col = mod(floor(abs(x3_elements(n+1:2*n)) * 1e8 ), 256);
    for j = 1:n
        if j == 1
            scrambled(:, j) = bitxor(scrambled(:, j), key_stream_col(j));
        else
            scrambled(:, j) = bitxor(scrambled(:, j), scrambled(:, j-1));
        end
    end
    
    % XOR encryption
    key_stream = mod(floor(abs(x3_elements)*1e4), 256);
    encrypted_block = bitxor(uint8(scrambled), uint8(reshape(key_stream, n, n)));
    
    % Update state with last pixel
    new_state = double(encrypted_block(end, end));
end

%% Block decryption function (double diffusion)
function decrypted_block = decrypt_block(encrypted_block, x1_i, x2_i, x3_elements, n, previous_state)
    % Generate row diffusion key using previous state
    key_stream_row = mod(floor( (abs(x3_elements(1:n)) .* x1_i + x2_i + previous_state) * 1e8 ), 256);
    
    % XOR decryption
    key_stream = mod(floor(abs(x3_elements)*1e4), 256);
    decrypted_scrambled = bitxor(uint8(encrypted_block), uint8(reshape(key_stream, n, n)));
    
    % Inverse column diffusion
    key_stream_col = mod(floor(abs(x3_elements(n+1:2*n)) * 1e8 ), 256);
    for j = n:-1:2
        decrypted_scrambled(:, j) = bitxor(decrypted_scrambled(:, j), decrypted_scrambled(:, j-1));
    end
    decrypted_scrambled(:, 1) = bitxor(decrypted_scrambled(:, 1), key_stream_col(1));
    
    % Inverse row diffusion
    for i = 1:n
        decrypted_scrambled(i, :) = bitxor(decrypted_scrambled(i, :), key_stream_row(i));
    end
    
    % Inverse scrambling
    start_row = mod(floor(abs(x1_i)*1e4), n) + 1;
    start_col = mod(floor(abs(x2_i)*1e4), n) + 1;
    seed = sum(floor(abs(x3_elements(1:4))*1e4));
    zigzag_order = generate_zigzag_sequence(n, [start_row, start_col], seed);
    [~, inv_order] = sort(zigzag_order(:));
    decrypted_block = block_zigzag_shuffle(double(decrypted_scrambled), inv_order);
end

%% Decryption image processing
function decrypted = decrypt_image(encrypted, x1_seq, x2_seq, x3_seq, n, M, N)
    encrypted_blocks = mat2cell(encrypted, n*ones(1,M/n), n*ones(1,N/n));
    decrypted_blocks = cell(size(encrypted_blocks));
    num_blocks = numel(encrypted_blocks);
    
    state = 0; % Initial state
    
    for i = 1:num_blocks
        encrypted_block = encrypted_blocks{i};
        x1_i = x1_seq(i);
        x2_i = x2_seq(i);
        x3_elements = x3_seq( (i-1)*(n*n)+1 : i*n*n );
        
        % Decrypt block using previous state
        decrypted_block = decrypt_block(encrypted_block, x1_i, x2_i, x3_elements, n, state);
        decrypted_blocks{i} = decrypted_block;
        
        % Update state with last encrypted pixel
        state = double(encrypted_block(end, end));
    end
    
    decrypted = cell2mat(decrypted_blocks);
end

%% Dynamic Zigzag sequence generation
function order = generate_zigzag_sequence(n, start_point, seed)
    % Set random seed for reproducibility
    rng(seed);
    
    order = zeros(n);
    visited = false(n);
    current_order = 1;
    
    % Initialize start point
    row = clamp(start_point(1), 1, n);
    col = clamp(start_point(2), 1, n);
    
    % Mark start point
    order(row, col) = current_order;
    visited(row, col) = true;
    current_order = current_order + 1;
    
    direction = 1; % Initial direction: 1=top-right, -1=bottom-left
    
    while current_order <= n*n
        % Try moving in current direction
        if direction == 1
            next_row = row - 1;
            next_col = col + 1;
        else
            next_row = row + 1;
            next_col = col - 1;
        end
        
        % Apply boundary constraints
        next_row = clamp(next_row, 1, n);
        next_col = clamp(next_col, 1, n);
        
        % Check validity
        if next_row >= 1 && next_row <= n && next_col >= 1 && next_col <= n && ~visited(next_row, next_col)
            row = next_row;
            col = next_col;
        else
            % Get all unvisited positions
            [unvisited_rows, unvisited_cols] = find(~visited);
            if isempty(unvisited_rows)
                break;
            end
            perm_idx = randperm(length(unvisited_rows), 1);
            row = unvisited_rows(perm_idx);
            col = unvisited_cols(perm_idx);
            
            % Random new direction
            direction = sign(rand() - 0.5);
            if direction == 0
                direction = 1;
            end
        end
        
        % Record position
        order(row, col) = current_order;
        visited(row, col) = true;
        current_order = current_order + 1;
    end
end

% Boundary constraint helper
function x = clamp(x, min_val, max_val)
    x = max(min_val, min(x, max_val));
end

%% Zigzag scrambling execution
function shuffled_block = block_zigzag_shuffle(block, order)
    [n, ~] = size(block);
    flat_block = block(:);
    [~, idx] = sort(order(:));
    shuffled_block = reshape(flat_block(idx), n, n);
end

%% Chaotic system
function dxdt = your_chaos_system(~, x)
    a = 8; b = 4; c = 14;
    dxdt = [a*x(1)-x(2)*x(3)+x(4);
            x(1)*x(3)-b*x(2)+3*x(3)-x(4).^3;
            x(1)*x(2)-c*x(3)+x(1);
            -2*x(4)-2*x(1)];
end

%% Hash function
function hash = get_image_hash(img)
    img_uint8 = uint8(img);
    img_bytes = typecast(img_uint8(:), 'uint8');
    hash_java = java.security.MessageDigest.getInstance('SHA-256');
    hash_java.update(img_bytes);
    hash = typecast(hash_java.digest(), 'uint8');
end

%% Initial conditions generation
function initials = hash_to_initial(hash, dim)
    hash_double = double(hash)/255;
    section_len = floor(length(hash)/dim);
    initials = zeros(dim,1);
    for i = 1:dim
        start_idx = (i-1)*section_len + 1;
        end_idx = i*section_len;
        segment = hash_double(start_idx:end_idx);
        initials(i) = sum(segment)/section_len;
    end
    initials = initials + 0.0001; % Avoid zero values
end

%% Visualization functions
function plot_histograms(original, encrypted, decrypted)
    figure('Name','Histogram Analysis');
    subplot(3,1,1), imhist(uint8(original)), title('Original');
    subplot(3,1,2), imhist(uint8(encrypted)), title('Encrypted');
    subplot(3,1,3), imhist(uint8(decrypted)), title('Decrypted');
end

function [chi2_result, pass] = chi2_test(encrypted_img)
    encrypted_img = uint8(round(encrypted_img));
    observed = imhist(encrypted_img);
    
    % Ensure full range
    if length(observed) < 256
        observed(256) = 0;
    end
    
    % Uniform distribution expectation
    expected = numel(encrypted_img) / 256 * ones(256,1);
    chi2_stat = sum((observed - expected).^2 ./ expected);

    % Critical value (df=255, α=0.05)
    alpha = 0.05;
    df = 255;
    critical_value = chi2inv(1 - alpha, df);
    pass = chi2_stat < critical_value;
    
    % Return result structure
    chi2_result = struct('Statistic', chi2_stat, 'Critical', critical_value, 'Pass', pass);
end

%% Correlation analysis
function corr_coeffs = correlation_analysis(img, direction)
    [~, N] = size(img);
    img = double(img(:));
    
    switch direction
        case 'horizontal'
            pairs = [img(1:end-N), img(N+1:end)]; % Exclude last row
        case 'vertical'
            pairs = [img(1:end-1), img(2:end)];    % Exclude last column
        case 'diagonal'
            pairs = [img(1:end-N-1), img(N+2:end)];% Exclude edges
    end
    
    corr_matrix = corrcoef(pairs);
    corr_coeffs = corr_matrix(1,2);
end

%% Key sensitivity test
function [npcr, uaci] = key_sensitivity_test(original_img, original_hash, modified_hash)
    if size(original_img,3) == 3
        original_img = rgb2gray(original_img);
    end
    original_img = double(original_img);
    
    % Encrypt with original hash
    initial_ori = hash_to_initial(original_hash, 4);
    encrypted_ori = encrypt_with_key(original_img, initial_ori);
    
    % Encrypt with modified hash
    initial_mod = hash_to_initial(modified_hash, 4);
    encrypted_mod = encrypt_with_key(original_img, initial_mod);
    
    % NPCR calculation
    diff_binary = (encrypted_ori ~= encrypted_mod);
    npcr = sum(diff_binary(:)) / numel(encrypted_ori) * 100;
    
    % UACI calculation
    diff_abs = abs(double(encrypted_ori) - double(encrypted_mod));
    uaci = sum(diff_abs(:)) / (numel(encrypted_ori) * 255) * 100;
end

%% Encryption wrapper
function encrypted = encrypt_with_key(img, initial_conditions)
    n = 8; % Match main function
    [M, N] = size(img);
    
    % Generate chaotic sequences
    num_blocks = (M/n) * (N/n);
    required_x1 = num_blocks;
    required_x2 = num_blocks;
    required_x3 = num_blocks * n*n;
    max_required = max([required_x1, required_x2, required_x3]);
    time_step = 0.01;
    time_vector = 0:time_step:(max_required * time_step);
    
    % Solve chaotic system
    [~, x] = ode45(@your_chaos_system, time_vector, initial_conditions);
    
    % Extract sequences
    x1_seq = x(1:required_x1, 1);
    x2_seq = x(1:required_x2, 2);
    x3_seq = x(1:required_x3, 3);
    
    % Process encryption
    encrypted = block_processing(img, x1_seq, x2_seq, x3_seq, n, M, N, @encrypt_block);
end

%% Information entropy
function entropy_value = image_entropy(img)
    img_uint8 = uint8(img);
    counts = imhist(img_uint8);
    prob = counts / sum(counts);
    entropy_value = -sum(prob(prob > 0) .* log2(prob(prob > 0)));
end

%% Correlation scatter plot
function plot_correlation_scatter(original, encrypted)
    sample_num = 2000;
    directions = {'Horizontal', 'Vertical', 'Diagonal'};
    figure('Position', [100 100 1200 800]);
    
    for i = 1:3
        % Original image
        subplot(3,2,2*i-1);
        [x_ori, y_ori] = get_pixel_pairs(original, directions{i});
        idx = randperm(length(x_ori), sample_num);
        scatter(x_ori(idx), y_ori(idx), 3, 'filled', 'MarkerFaceAlpha',0.3);
        axis([0 250 0 250]);
        title(sprintf('Original %s (r=%.3f)', directions{i}, ...
            correlation_analysis(original, lower(directions{i}))));
        
        % Encrypted image
        subplot(3,2,2*i);
        [x_enc, y_enc] = get_pixel_pairs(encrypted, directions{i});
        idx = randperm(length(x_enc), sample_num);
        scatter(x_enc(idx), y_enc(idx), 3, 'filled', 'MarkerFaceAlpha',0.3);
        axis([0 250 0 250]);
        title(sprintf('Encrypted %s (r=%.3f)', directions{i}, ...
            correlation_analysis(encrypted, lower(directions{i}))));
    end
end

% Pixel pair extraction helper
function [x, y] = get_pixel_pairs(img, direction)
    img = double(img);
    [~, ~] = size(img);
    
    switch lower(direction)
        case 'horizontal'
            x = img(1:end, 1:end-1);
            y = img(1:end, 2:end);
        case 'vertical'
            x = img(1:end-1, 1:end);
            y = img(2:end, 1:end);
        case 'diagonal'
            x = img(1:end-1, 1:end-1);
            y = img(2:end, 2:end);
    end
    x = x(:);
    y = y(:);
end

%% Noise attack analysis
function [noisy_encrypted, decrypted, psnr_val] = noise_attack_analysis(encrypted, x1_seq, x2_seq, x3_seq, n, M, N, original_img, noise_density)
    % Add salt & pepper noise
    noisy_encrypted = imnoise(uint8(encrypted), 'salt & pepper', noise_density);
    
    % Decrypt noisy image
    decrypted = decrypt_image(noisy_encrypted, x1_seq, x2_seq, x3_seq, n, M, N);
    decrypted = arnold_unscramble(decrypted, 50);
    
    % Calculate PSNR
    psnr_val = psnr(uint8(original_img), uint8(decrypted));
end

%% Cropping attack analysis
function [cropped_results] = crop_attack_analysis(encrypted, x1_seq, x2_seq, x3_seq, n, M, N, original_img)
    cropped_results = struct();

    %% 1. Center rectangle crop
    crop_size = floor([M/4, N/4]);
    start_pos = floor([M/2 - crop_size(1)/2, N/2 - crop_size(2)/2]);
    cropped_rect = encrypted;
    cropped_rect(start_pos(1):start_pos(1)+crop_size(1), start_pos(2):start_pos(2)+crop_size(2)) = 0;
    decrypted_rect = decrypt_image(cropped_rect, x1_seq, x2_seq, x3_seq, n, M, N);
    decrypted_rect = arnold_unscramble(decrypted_rect, 50);
    psnr_rect = psnr(uint8(original_img), uint8(decrypted_rect));
    cropped_results.rectangle_center = struct('image', cropped_rect, 'decrypted', decrypted_rect, 'psnr', psnr_rect);

    %% 2. Center circle crop
    [h, w] = size(encrypted);
    radius = floor(min(h,w)*0.3);
    [X,Y] = meshgrid(1:w,1:h);
    mask = (X-w/2).^2 + (Y-h/2).^2 <= radius^2;
    cropped_circle = encrypted;
    cropped_circle(mask) = 0;
    decrypted_circle = decrypt_image(cropped_circle, x1_seq, x2_seq, x3_seq, n, M, N);
    decrypted_circle = arnold_unscramble(decrypted_circle, 50);
    psnr_circle = psnr(uint8(original_img), uint8(decrypted_circle));
    cropped_results.circle_center = struct('image', cropped_circle, 'decrypted', decrypted_circle, 'psnr', psnr_circle);

    %% 3. Edge crop
    edge_width = floor(N/6);
    cropped_edge = encrypted;
    cropped_edge(:, 1:edge_width) = 0;
    decrypted_edge = decrypt_image(cropped_edge, x1_seq, x2_seq, x3_seq, n, M, N);
    decrypted_edge = arnold_unscramble(decrypted_edge, 50);
    psnr_edge = psnr(uint8(original_img), uint8(decrypted_edge));
    cropped_results.edge = struct('image', cropped_edge, 'decrypted', decrypted_edge, 'psnr', psnr_edge);

    %% 4. Band crop
    band_width = floor(h*0.1);
    cropped_band = encrypted;
    cropped_band(1:band_width, :) = 0;       % Top edge
    cropped_band(end-band_width:end, :) = 0; % Bottom edge
    cropped_band(:, 1:band_width) = 0;       % Left edge
    cropped_band(:, end-band_width:end) = 0; % Right edge
    decrypted_band = decrypt_image(cropped_band, x1_seq, x2_seq, x3_seq, n, M, N);
    decrypted_band = arnold_unscramble(decrypted_band, 50);
    psnr_band = psnr(uint8(original_img), uint8(decrypted_band));
    cropped_results.band = struct('image', cropped_band, 'decrypted', decrypted_band, 'psnr', psnr_band);
end

%% Arnold scrambling function
function scrambled_img = arnold_scramble(img, iterations)
    [M, N] = size(img);
    scrambled_img = img;
    for iter = 1:iterations
        temp = zeros(M, N);
        for x = 1:M
            for y = 1:N
                % Arnold transform: [x'; y'] = [1 1; 1 2] * [x-1; y-1] mod M/N
                new_x = mod((x-1) + (y-1), M) + 1;
                new_y = mod((x-1) + 2*(y-1), N) + 1;
                temp(new_x, new_y) = scrambled_img(x, y);
            end
        end
        scrambled_img = temp;
    end
end

%% Inverse Arnold scrambling
function unscrambled_img = arnold_unscramble(img, iterations)
    [M, N] = size(img);
    unscrambled_img = img;
    for iter = 1:iterations
        temp = zeros(M, N);
        for x = 1:M
            for y = 1:N
                % Inverse transform: [x'; y'] = [2 -1; -1 1] * [x-1; y-1] mod M/N
                new_x = mod(2*(x-1) - (y-1), M) + 1;
                new_y = mod(-(x-1) + (y-1), N) + 1;
                temp(new_x, new_y) = unscrambled_img(x, y);
            end
        end
        unscrambled_img = temp;
    end
end

%% Sensitivity calculation
function [npcr, uaci] = calculate_sensitivity(C1, C2)
    C1 = double(C1);
    C2 = double(C2);
    [M, N] = size(C1);
    
    % NPCR calculation
    diff_pixels = sum(C1(:) ~= C2(:));
    npcr = (diff_pixels / (M*N)) * 100;
    
    % UACI calculation
    abs_diff = sum(abs(C1(:) - C2(:)));
    max_intensity = 255;  % 8-bit grayscale
    uaci = (abs_diff / (M*N*max_intensity)) * 100;
end

%% Differential attack test
function [npcr, uaci] = differential_attack_test(original_img)
    if size(original_img,3) == 3
        original_img = rgb2gray(original_img);
    end
    original_img = double(original_img);

    % Original image hash
    hash_original = get_image_hash(original_img);
    initial_ori = hash_to_initial(hash_original, 4);
    encrypted_ori = encrypt_with_key(original_img, initial_ori);

    % Single-pixel modification
    disturbed_img = original_img;
    disturbed_img(1,1) = mod(disturbed_img(1,1) + 1, 256);

    % Modified image hash
    hash_disturbed = get_image_hash(disturbed_img);
    initial_disturbed = hash_to_initial(hash_disturbed, 4);
    encrypted_disturbed = encrypt_with_key(disturbed_img, initial_disturbed);

    % NPCR
    diff_binary = (encrypted_ori ~= encrypted_disturbed);
    npcr = sum(diff_binary(:)) / numel(encrypted_ori) * 100;

    % UACI
    diff_abs = abs(double(encrypted_ori) - double(encrypted_disturbed));
    uaci = sum(diff_abs(:)) / (numel(encrypted_ori) * 255) * 100;
end