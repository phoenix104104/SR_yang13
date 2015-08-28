function [feature_all, mean_all] = ...
                extract_LR_cnn_feature_from_position(img_LR, cnn_net, ...
                                                     parameter, position)
    
    img_to_cnn = single(imresize(img_LR, cnn_net.normalization.imageSize(1:2)));
    img_to_cnn = img_to_cnn - cnn_net.normalization.averageImage;
    cnn_LR = extract_cnn_featuremap(cnn_net, img_to_cnn, parameter.cnn_target_layer);
    cnn_LR = imresize(cnn_LR, [size(img_LR, 1), size(img_LR, 2)]);
        
    if( size(img_LR, 3) == 3 )
        img_LR = RGB2Y(img_LR);
    end

    
    patch_size      = parameter.LR_patch_size;
    half_patch_size = (patch_size - 1) / 2;
    
    exclude_corner_mask = parameter.exclude_corner_mask;
    feature_dimension   = parameter.LR_feature_dimension;
    
    cnn_layer = size(cnn_LR, 3);
    cnn_extract_mask = exclude_corner_mask;
    offset = patch_size * patch_size;
    for i = 1:cnn_layer-1
        cnn_extract_mask = [cnn_extract_mask; ...
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
        
        
        cnn_patch = cnn_LR(r-half_patch_size:r+half_patch_size, ...
                           c-half_patch_size:c+half_patch_size, :);
        feature = cnn_patch(cnn_extract_mask);
        feature_all(i, :) = feature';
        
    end
    
end

