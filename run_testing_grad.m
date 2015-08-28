function run_testing_grad(config_filename)

fprintf('load %s\n', config_filename);
eval(config_filename);

addpath('../../util');
addpath('../../util/metrix_mux');

method = sprintf('yang13_%s_data%d_grad_LR%d_HR%d_corner%d_center%d_nn%d_lambda%s', ...
                  parameter.training_dataset, ...
                  parameter.num_patches_for_kmeans, ...
                  parameter.LR_patch_size, ...
                  parameter.HR_patch_size, ...
                  parameter.exclude_corner, ...
                  parameter.num_cluster, ...
                  parameter.num_patches_for_regressor, ...
                  num2str(parameter.lambda));


dataset = 'BSD200';
scale = parameter.scaling_factor;
sigma = parameter.sigma;
LR_dir = fullfile(sprintf('sf%s', num2str(scale)), sprintf('sigma%s', num2str(sigma)));

image_dir = fullfile('../../images', dataset);

output_dir = fullfile(image_dir, LR_dir, method);
if( ~exist(output_dir, 'dir') )
    mkdir(output_dir);
end

configure_metrix_mux;
x_offset = scale; y_offset = scale; search_range = scale;

list_name = sprintf('../../list/%s.txt', dataset);
file_list = load_list(list_name);
img_num = length(file_list);


%% Load cluster centers
model_dir = fullfile('training', LR_dir, 'cluster_centers');
center_filename = fullfile(model_dir, ...
                           sprintf('cluster_centers_%s_data%d_grad_LR%d_HR%d_corner%d_center%d.mat', ...
                           parameter.training_dataset, ...
                           parameter.num_patches_for_kmeans, ...
                           parameter.LR_patch_size, ...
                           parameter.HR_patch_size, ...
                           parameter.exclude_corner, ...
                           parameter.num_cluster));

fprintf('Load %s\n', center_filename);
data = load(center_filename);
cluster_centers = data.cluster_centers;
clear data;

%% Load regressors
model_dir = fullfile('training', LR_dir, 'regressors');
regressor_filename = fullfile(model_dir, ...
                              sprintf('regressors_%s_data%d_grad_LR%d_HR%d_corner%d_center%d_nn%d_lambda%s.mat', ...
                              parameter.training_dataset, ...
                              parameter.num_patches_for_kmeans, ...
                              parameter.LR_patch_size, ...
                              parameter.HR_patch_size, ...
                              parameter.exclude_corner, ...
                              parameter.num_cluster, ...
                              parameter.num_patches_for_regressor, ...
                              num2str(parameter.lambda)));
                          
fprintf('Load %s\n', regressor_filename);
data = load(regressor_filename);
regressors = data.regressors;
clear data;


%% Load PCA projection matrix
model_dir = fullfile('training', LR_dir, 'pca');
pca_filename = fullfile(model_dir, ...
                        sprintf('pca_%s_data%d_grad_LR%d_HR%d_corner%d.mat', ...
                        parameter.training_dataset, ...
                        parameter.num_patches_for_kmeans, ...
                        parameter.LR_patch_size, ...
                        parameter.HR_patch_size, ...
                        parameter.exclude_corner));

fprintf('Load %s\n', pca_filename);
data = load(pca_filename);
parameter.M_pca = data.M_pca;
clear data;



%% reconstruct HR images
t_all = zeros(img_num, 1);
PSNR  = zeros(img_num, 1);
SSIM  = zeros(img_num, 1);
IFC   = zeros(img_num, 1);

for i = 1:img_num
    % Load HR image
    img_name = fullfile(image_dir, 'GroundTruth', sprintf('%s.bmp', file_list{i}));
    img_GT = im2double(imread(img_name));
    
    % Load LR image
    img_name = fullfile(image_dir, LR_dir, 'input', sprintf('%s.bmp', file_list{i}));
    fprintf('Process image %d: %s\n', i, img_name);
    img_LR = im2double(imread(img_name));
    
    % RGB to YIQ
    img_yiq_lr = RGB2YIQ(img_LR);
    
    % bicubic interpolation
    img_yiq_hr = imresize(img_yiq_lr, scale, 'bicubic');
    
    % upscale on Y
    img_y = img_yiq_lr(:, :, 1);
    
    tic;
    img_yiq_hr(:, :, 1) = SR_simple_function_grad(img_LR, parameter, cluster_centers, regressors);
    t_all(i) = toc;
    
    % YIQ to RGB
	img_rgb_hr = YIQ2RGB(img_yiq_hr);
    
    % save image
    filename = fullfile(output_dir, sprintf('%s.bmp', file_list{i}));
    fprintf('Save %s\n', filename);
    imwrite(img_rgb_hr, filename);
    %figure, imshow(img_rgb_hr); drawnow; pause;

    % evaluate PSNR, SSIM, IFC
    img_hr = img_rgb_hr(scale+1:end-scale, scale+1:end-scale, :);
    img_gt = align_and_crop(img_hr, img_GT, search_range, x_offset, y_offset);
    [PSNR(i), SSIM(i)] = F7_ComputePSNR_SSIM(img_gt, img_hr);
    IFC(i) = metrix_mux(img_gt, img_hr, 'ifc');
    
end

fprintf('Average running time = %f sec\n', mean(t_all));
fprintf('average PSNR = %f\n', mean(PSNR));
fprintf('average SSIM = %f\n', mean(SSIM));
fprintf('average IFC = %f\n', mean(IFC));


filename = fullfile(output_dir, 'PSNR.txt');
save_matrix(PSNR, filename);

filename = fullfile(output_dir, 'SSIM.txt');
save_matrix(SSIM, filename);

filename = fullfile(output_dir, 'IFC.txt');
save_matrix(IFC, filename);

end
