%% define parameters
config_22;

addpath('../../util');
run ../../util/vlfeat-0.9.20/toolbox/vl_setup;

input_dir    = fullfile('../../images', parameter.training_dataset);
LR_dir       = fullfile(sprintf('sf%s', num2str(parameter.scaling_factor)), ...
                        sprintf('sigma%s', num2str(parameter.sigma)));
training_dir = fullfile('training', LR_dir);

filename = fullfile('../../list', sprintf('%s_ext.txt', parameter.training_dataset));
fprintf('Load %s\n', filename);
training_filelist = load_list(filename);

num_images = length(training_filelist);

%% Load HR images and collect training patches for kmeans
output_dir = fullfile(training_dir, 'kmeans_features');
if( ~exist(output_dir, 'dir') )
    mkdir(output_dir);
end
feature_filename = fullfile(output_dir, ...
                            sprintf('kmeans_features_%s_data%d_LR%d_HR%d_corner%d.mat', ...
                            parameter.training_dataset, ...
                            parameter.num_patches_for_kmeans, ...
                            parameter.LR_patch_size, ...
                            parameter.HR_patch_size, ...
                            parameter.exclude_corner));
tic;                 
if( exist(feature_filename, 'file') )
    
    fprintf('Load %s\n', feature_filename);
    data = load(feature_filename);
    feature_all = data.feature_all;
    clear data;
    
else
    
    fprintf('Load all HR images (Total %d images)...\n', num_images);
    HR_img_all = cell(num_images, 1);

    for i = 1:num_images
        filename = fullfile(input_dir, training_filelist{i});
        HR_img_all{i} = imread(filename);
        %fprintf('Process image %d: %s\n', i, filename);
    end

    fprintf('Collect training patches for kmeans...\n');
    num_patch_per_image = round(parameter.num_patches_for_kmeans / num_images);

    feature_all = [];
    downsample_factor = 1;

    while( true )

        for i = 1:num_images
            %fprintf('Process image %d\n', i);

            HR_img = imresize(im2double(HR_img_all{i}), downsample_factor);
            LR_img = GenerateLRImage(HR_img, parameter.scaling_factor, parameter.sigma);
            [feature, position] = extract_LR_feature(LR_img, parameter, ...
                                                     num_patch_per_image);
            feature_all = [feature_all; feature];

            if( size(feature_all, 1) >= parameter.num_patches_for_kmeans )
                fprintf('Collect %d sample patches!\n', parameter.num_patches_for_kmeans);
                break;
            end
        end

        if( size(feature_all, 1) < parameter.num_patches_for_kmeans )
            % if not enough samples, downscale and extract patches again
            downsample_factor = downsample_factor * parameter.collect_downsample_factor;
            fprintf('Not enough patches (%d/%d), downsample HR images with %sx\n', ...
                    size(feature_all, 1), parameter.num_patches_for_kmeans, ...
                    num2str(downsample_factor));
        else
            break;
        end
    end

    feature_all = feature_all(1:parameter.num_patches_for_kmeans, :);
    fprintf('Save %s\n', feature_filename);
    save(feature_filename, 'feature_all');
    
    clear HR_images;
end
t_collect_kmeans_feature = toc;


%% kmeans clustering
output_dir = fullfile(training_dir, 'cluster_centers');
if( ~exist(output_dir, 'dir') )
    mkdir(output_dir);
end
center_filename = fullfile(output_dir, ...
                           sprintf('cluster_centers_%s_data%d_LR%d_HR%d_corner%d_center%d.mat', ...
                           parameter.training_dataset, ...
                           parameter.num_patches_for_kmeans, ...
                           parameter.LR_patch_size, ...
                           parameter.HR_patch_size, ...
                           parameter.exclude_corner, ...
                           parameter.num_cluster));

tic;
if( exist(center_filename, 'file') )
    
    fprintf('Load %s\n', center_filename);
    data = load(center_filename);
    cluster_centers = data.cluster_centers;
    clear data;
    
else

    fprintf('k-means clustering (%d centers)\n', parameter.num_cluster);

    max_iteration = 100;
    opts = statset('Display', 'iter', 'MaxIter', max_iteration);
    tic
    [idx, centers] = kmeans(feature_all, parameter.num_cluster, ...
                      'emptyaction', 'drop', 'options',opts);
    toc
    
    clear feature_all;
    
    % calculate #samples for each center
    num_sample = hist(idx, parameter.num_cluster);
    % sorted centers by #samples
    [~,idx_sort] = sort(num_sample, 'descend');
    cluster_centers = centers(idx_sort, :);
    
    fprintf('Save %s\n', center_filename);
    save(center_filename, 'cluster_centers');
    
end
t_kmeans = toc;


%% build kd-tree
% fprintf('Build kdtree...\n');
% kdtree = vl_kdtreebuild(cluster_centers');


%% extract label(nearest cluster center) for all LR patches
output_dir = fullfile(training_dir, 'label_position');
if( ~exist(output_dir, 'dir') )
    mkdir(output_dir);
end
label_filename = fullfile(output_dir, ...
                          sprintf('label_position_%s_data%d_LR%d_HR%d_corner%d_center%d.mat', ...
                          parameter.training_dataset, ...
                          parameter.num_patches_for_kmeans, ...
                          parameter.LR_patch_size, ...
                          parameter.HR_patch_size, ...
                          parameter.exclude_corner, ...
                          parameter.num_cluster));
       
tic;
if( exist(label_filename, 'file') )
    
    fprintf('Load %s\n', label_filename);
    data = load(label_filename);
    label_all       = data.label_all;
    LR_position_all = data.LR_position_all;
    clear data;
    
else

    fprintf('Extract labels for LR patches (Total %d images)...\n', num_images);
    label_all = cell(num_images, 1);
    LR_position_all = cell(num_images, 1);
    
    for i = 1:num_images
        filename = fullfile(input_dir, training_filelist{i});
        HR_img = im2double(imread(filename));
        LR_img = GenerateLRImage(HR_img, parameter.scaling_factor, parameter.sigma);
        [LR_features, LR_position] = extract_LR_feature(LR_img, parameter);
        idx = knnsearch(cluster_centers, LR_features);
        %idx = vl_kdtreequery(kdtree, cluster_centers', LR_features');
        label_all{i} = uint32(idx);
        LR_position_all{i} = uint32(LR_position);
    end
    
    fprintf('Save %s\n', label_filename);
    save(label_filename, 'label_all', 'LR_position_all', '-v7.3');
    
end
t_label = toc;
    
%% compute regressors
output_dir = fullfile(training_dir, 'regressors');
if( ~exist(output_dir, 'dir') )
    mkdir(output_dir);
end

parameter_tag = sprintf('%s_data%d_LR%d_HR%d_corner%d_center%d_nn%d', ...
                        parameter.training_dataset, ...
                        parameter.num_patches_for_kmeans, ...
                        parameter.LR_patch_size, ...
                        parameter.HR_patch_size, ...
                        parameter.exclude_corner, ...
                        parameter.num_cluster, ...
                        parameter.num_patches_for_regressor);

regressor_filename = fullfile(output_dir, sprintf('regressors_%s_lambda%s.mat', ...
                                          parameter_tag, ...
                                          num2str(parameter.lambda)));

output_dir = fullfile(training_dir, 'regressors', parameter_tag);
if( ~exist(output_dir, 'dir') )
    mkdir(output_dir);
end

tic;
if( exist(regressor_filename, 'file') )
    
    fprintf('Load %s\n', regressor_filename);
    data = load(regressor_filename);
    regressors = data.regressors;
    clear data;
    
else

    fprintf('Train mapping function for each cluster...\n');
    regressors = cell(parameter.num_cluster, 1);

    for c = 1:parameter.num_cluster

        feature_filename = fullfile(output_dir, ...
                                    sprintf('regressor_features_%s_c%d.mat', ...
                                    parameter_tag, c));


        if( exist(feature_filename, 'file') )

            fprintf('Load %s\n', feature_filename);
            data = load(feature_filename);
            L = data.L;
            H = data.H;
            clear data;

        else
                    
            H = [];
            L = [];
            
            idx_rand = randperm(num_images);
            for i = 1:num_images
                idx = idx_rand(i);
                
                filename = fullfile(input_dir, training_filelist{idx});
                HR_img = im2double(imread(filename));
            
                label = label_all{idx};
                label_mask = (label == c);
                LR_position = LR_position_all{idx}(label_mask, :);
                [LR_features, HR_features] = extract_HR_feature_from_position(...
                                                HR_img, parameter, LR_position);
                
                L = [L; LR_features];
                H = [H; HR_features];
    
                if( size(L, 1) >= parameter.num_patches_for_regressor )
                    L = L(1:parameter.num_patches_for_regressor, :);
                    H = H(1:parameter.num_patches_for_regressor, :);
                    break;
                end
            end
            
            fprintf('Cluster %d collects %d samples.\n', c, size(L, 1));
            
            if( parameter.save_regressor_feature )
                fprintf('Save %s\n', feature_filename);
                save(feature_filename, 'L', 'H', '-v7.3');
            end

        end


        % add bias term
        L = [L, ones(size(L, 1), 1)];

        % regularization term
        I = eye(parameter.LR_feature_dimension + 1); 

        % no regularization on bias term
        I(end, end) = 0; 

        % W = L\H is not stable
        W = (L' * L + parameter.lambda * I) \ (L' * H); % W in D_LR x D_HR
        regressors{c} = W';
    end
    
    fprintf('Save %s\n', regressor_filename);
    save(regressor_filename, 'regressors');
    
end
t_regressor = toc;


%% Report running time
fprintf('==================================================\n');
fprintf('Collect kmeans features: %s seconds.\n', num2str(t_collect_kmeans_feature));
fprintf('kmeans clustering      : %s seconds.\n', num2str(t_kmeans));
fprintf('Extract labels         : %s seconds.\n', num2str(t_label));
fprintf('Calculate regressors   : %s seconds.\n', num2str(t_regressor));
fprintf('==================================================\n');
