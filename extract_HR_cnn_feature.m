function [LR_features, HR_features, position_all] = extract_HR_cnn_feature(img_HR, cnn_net, parameter, num_patch)
    
    if( ~exist('num_patch', 'var') )
        num_patch = [];
    end
    
    img_LR = GenerateLRImage(img_HR, parameter.scaling_factor, parameter.sigma);
    
    [LR_features, position_all, mean_all] = extract_LR_cnn_feature(img_LR, cnn_net, parameter, num_patch);
    
    if( size(img_HR, 3) == 3 )
        img_HR = RGB2Y(img_HR);
    end

    num_feature = size(LR_features, 1);
    HR_features = zeros(num_feature, parameter.HR_feature_dimension);
    
    for i = 1:num_feature
        LR_r = position_all(i, 1);
        LR_c = position_all(i, 2);
        
        HR_r1 = (LR_r - 2) * parameter.scaling_factor + 1;
        HR_r2 = (LR_r + 1) * parameter.scaling_factor;
        HR_c1 = (LR_c - 2) * parameter.scaling_factor + 1;
        HR_c2 = (LR_c + 1) * parameter.scaling_factor;
        
        patch = img_HR(HR_r1:HR_r2, HR_c1:HR_c2);
        feature = patch(:) - mean_all(i);
        HR_features(i, :) = feature';
    end
    
end
