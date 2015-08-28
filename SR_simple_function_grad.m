function img_HR = SR_simple_function_grad(img_LR, parameter, cluster_centers, regressors)
    
    M_pca   = parameter.M_pca; 
    sf      = parameter.scaling_factor;
    
    LR_patch_size = parameter.LR_patch_size;
    HR_patch_size = parameter.HR_patch_size;
    num_pixel_HR = (HR_patch_size)^2;
    half_patch_size = (LR_patch_size - 1) / 2;
    
    %% extend image boundary
    img_LR_ext = wextend('2d','symw', img_LR, half_patch_size);
    [h_LR_ext, w_LR_ext, ~] = size(img_LR_ext);
    
    %% extract LR features
    [LR_feature, LR_position, LR_mean, smooth_mask] = ...
                    extract_LR_grad_feature(img_LR_ext, parameter);
    
    LR_feature = LR_feature * M_pca;
	num_patch_LR = size(LR_feature, 1);
	
    img_LR_ext = RGB2Y(img_LR_ext);
    
    
    % calculate boundary mask
    x_st = half_patch_size + 1;
    x_ed = w_LR_ext - half_patch_size;
    y_st = half_patch_size + 1;
    y_ed = h_LR_ext - half_patch_size;
    
    boundary_mask = ones(h_LR_ext, w_LR_ext);
    boundary_mask(y_st:y_ed, x_st:x_ed) = 0;
    smooth_mask(boundary_mask == 1) = 0;
    
    [col, row] = meshgrid(1:w_LR_ext, 1:h_LR_ext);
    col_smooth = col(smooth_mask == 1);
    row_smooth = row(smooth_mask == 1);
    
    num_patch_smooth = length(col_smooth);
    
    %% find cluster center
    
    % kdtree query
    %label_LR = vl_kdtreequery(kdtree, cluter_centers', feature_LR');
    
    % knnsearch
    label_LR = knnsearch(cluster_centers, LR_feature, 'NSmethod', 'exhaustive');

        
    %% apply SR projection matrix
    predict_HR_feature = zeros(num_patch_LR, num_pixel_HR);
    
    for i = 1:num_patch_LR
        
        cluster_idx = label_LR(i);
        
        vector_LR = [LR_feature(i, :)'; 1];
        vector_HR = regressors{cluster_idx} * vector_LR;
        
        predict_HR_feature(i, :) = vector_HR' + LR_mean(i);
        
    end
    predict_HR_feature(predict_HR_feature > 1) = 1;
    predict_HR_feature(predict_HR_feature < 0) = 0;

    
    %% bicubic interpolate for smooth region
    img_HR_bicubic = imresize(img_LR_ext, sf, 'bicubic');
    [h_HR_ext, w_HR_ext] = size(img_HR_bicubic);
    
    
    %% reconstruct HR image from predicted patches
    img_HR_ext_sum   = zeros(h_HR_ext, w_HR_ext);
    img_HR_ext_count = zeros(h_HR_ext, w_HR_ext);
    
    % the algorithm only recover the central (3*sf) *(3*sf) in HR, so there is a offset in HR
    offset = 2 * sf;
    for i = 1:num_patch_LR
        
        % (LR_r, LR_c) is the top-left corner of a patch
        LR_r = LR_position(i, 1) - half_patch_size;
        LR_c = LR_position(i, 2) - half_patch_size;

        r1 = ( (LR_r - 1) * sf + 1 ) + offset;
        r2 = r1 + HR_patch_size - 1;
        
        c1 = ( (LR_c - 1) * sf + 1 ) + offset;
        c2 = c1 + HR_patch_size - 1;
        
        HR_patch = reshape(predict_HR_feature(i, :), [HR_patch_size, HR_patch_size]);
        
        img_HR_ext_sum(r1:r2, c1:c2)   = img_HR_ext_sum(r1:r2, c1:c2) + HR_patch;
        img_HR_ext_count(r1:r2, c1:c2) = img_HR_ext_count(r1:r2, c1:c2) + 1;
        
    end
    
    
    %% reconstruct smooth region from bicubic interpolation
    for i = 1:num_patch_smooth
        
        % (LR_r, LR_c) is the top-left corner of a patch
        LR_r = row_smooth(i) - half_patch_size;
        LR_c = col_smooth(i) - half_patch_size;

        r1 = ( (LR_r - 1) * sf + 1 ) + offset;
        r2 = r1 + HR_patch_size - 1;
        
        c1 = ( (LR_c - 1) * sf + 1 ) + offset;
        c2 = c1 + HR_patch_size - 1;
        
        HR_patch = img_HR_bicubic(r1:r2, c1:c2);
        
        img_HR_ext_sum(r1:r2, c1:c2)   = img_HR_ext_sum(r1:r2, c1:c2) + HR_patch;
        img_HR_ext_count(r1:r2, c1:c2) = img_HR_ext_count(r1:r2, c1:c2) + 1;
        
    end
    
    %% average all patches
    img_HR_ext_avg = img_HR_ext_sum ./ img_HR_ext_count;
    HR_ext_boundary = half_patch_size * sf;
    
    img_HR = img_HR_ext_avg(HR_ext_boundary+1:end-HR_ext_boundary ...
                           ,HR_ext_boundary+1:end-HR_ext_boundary);

end
