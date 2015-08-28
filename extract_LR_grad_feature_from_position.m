function [feature_all, mean_all] = ...
                extract_LR_grad_feature_from_position(img_LR, parameter, position)
    
            
    if( size(img_LR, 3) == 3 )
        img_LR = RGB2Y(img_LR);
    end

    grad_map = apply_grad_filters(img_LR, parameter.filters);

    patch_size      = parameter.LR_patch_size;
    half_patch_size = (patch_size - 1) / 2;
    
    exclude_corner_mask = parameter.exclude_corner_mask;
    feature_dimension   = parameter.LR_feature_dimension;
    
    feature_extract_mask = exclude_corner_mask;
    offset = patch_size * patch_size;
    for i = 1:size(grad_map, 3)-1
        feature_extract_mask = [feature_extract_mask; ...
                                offset * i + exclude_corner_mask];
    end
    
    num_patch = size(position, 1);
    
    feature_all  = zeros(num_patch, feature_dimension);
    mean_all     = zeros(num_patch, 1);
    
    for i = 1:num_patch
        r = position(i, 1);
        c = position(i, 2);
        
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

