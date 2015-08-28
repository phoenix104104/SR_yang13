function [feature_all, mean_all] = ...
                extract_LR_feature_from_position(img_LR, parameter, position)
    
    if( size(img_LR, 3) == 3 )
        img_LR = RGB2Y(img_LR);
    end

    
    patch_size      = parameter.LR_patch_size;
    half_patch_size = (patch_size - 1) / 2;
    
    exclude_corner_mask = parameter.exclude_corner_mask;
    feature_dimension   = parameter.LR_feature_dimension;
    
    
    num_patch = size(position, 1);
    
    feature_all  = zeros(num_patch, feature_dimension);
    mean_all     = zeros(num_patch, 1);
    
    for i = 1:num_patch
        r = position(i, 1);
        c = position(i, 2);
        
        patch = img_LR(r-half_patch_size:r+half_patch_size, ...
                       c-half_patch_size:c+half_patch_size);
        feature = patch(exclude_corner_mask);
        mu = mean(feature);
        feature = feature - mu;
        feature_all(i, :) = feature';
        mean_all(i) = mu;
    end
    
end

