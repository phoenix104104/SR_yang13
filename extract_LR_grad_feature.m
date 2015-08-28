function [feature_all, position_all, mean_all, ignore_mask] = ...
                extract_LR_grad_feature(img_LR, parameter, num_patch)
    
    
    if( size(img_LR, 3) == 3 )
        img_LR = RGB2Y(img_LR);
    end
    
    grad_map = apply_grad_filters(img_LR, parameter.filters);
    
    patch_size      = parameter.LR_patch_size;
    half_patch_size = (patch_size - 1) / 2;
    
    exclude_corner_mask = parameter.exclude_corner_mask;
    
    feature_extract_mask = exclude_corner_mask;
    offset = patch_size * patch_size;
    for i = 1:size(grad_map, 3)-1
        feature_extract_mask = [feature_extract_mask; ...
                                offset * i + exclude_corner_mask];
    end
    
    feature_dimension = length(feature_extract_mask);
    
    [h, w] = size(img_LR);
    
    % calculate boundary mask
    x_st = half_patch_size + 1;
    x_ed = w - half_patch_size;
    y_st = half_patch_size + 1;
    y_ed = h - half_patch_size;
    
    boundary_mask = ones(h, w);
    boundary_mask(y_st:y_ed, x_st:x_ed) = 0;
    
    ignore_mask = calculate_smooth_mask(img_LR, parameter);
    ignore_mask(boundary_mask == 1) = 1;
    
    [col_all, row_all] = meshgrid(1:w, 1:h);
    col_all = col_all(ignore_mask == 0);
    row_all = row_all(ignore_mask == 0);
    
    total_num_patch = sum(ignore_mask(:) == 0);
    if( ~exist('num_patch', 'var') )
        num_patch = total_num_patch;
    end
    
    if( num_patch < total_num_patch )
        patch_index = randperm(total_num_patch, num_patch);
    else
        %fprintf('Require more patches...\n');
        num_patch = total_num_patch; % don't allow required #patches > total #patches
        patch_index = 1:total_num_patch;
    end
    
    feature_all  = zeros(num_patch, feature_dimension);
    position_all = zeros(num_patch, 2); %[row, col]
    mean_all     = zeros(num_patch, 1);
    
    for i = 1:length(patch_index)
        index = patch_index(i);
        r = row_all(index);
        c = col_all(index);
        position_all(i, :) = [r, c];
        
        
        patch = img_LR(r-half_patch_size:r+half_patch_size, ...
                       c-half_patch_size:c+half_patch_size);
        mu = mean(patch(exclude_corner_mask));
        mean_all(i) = mu;
        
        
        grad_patch = grad_map(r-half_patch_size:r+half_patch_size, ...
                              c-half_patch_size:c+half_patch_size, :);
        feature = grad_patch(feature_extract_mask);
        feature_all(i, :) = feature';
        
    end
    
end

