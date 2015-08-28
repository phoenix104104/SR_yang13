function smooth_mask = calculate_smooth_mask(img_LR, parameter)
    
    smooth_radius   = parameter.smooth_radius;
	gradient_thr    = parameter.gradient_thr;
    num_smooth_thr  = parameter.num_smooth_thr;
    
    % calculate 8-D gradient maps
    gradient_lr = F14c_Img2Grad_fast_suppressboundary(img_LR);
    
    % mask indicated smooth region
    smooth_mask = double(abs(gradient_lr) <= gradient_thr);
    
    % box filter == sum up value in each patch
    for i = 1:size(smooth_mask, 3)
        smooth_mask(:, :, i) = boxfilter(smooth_mask(:, :, i), smooth_radius);
    end
    smooth_mask = sum(smooth_mask, 3);
    
    % patch in this mask will be ignored
    smooth_mask = smooth_mask >= num_smooth_thr;
    
end