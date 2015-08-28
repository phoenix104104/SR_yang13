function corner_mask = find_exclude_corner_mask(parameter)
    
    patch_size = parameter.LR_patch_size;
    % calculate the index of feature (exclude corner or not)
    if( parameter.exclude_corner )
        [x, y] = meshgrid(1:patch_size, 1:patch_size);
        mask = ones(patch_size, patch_size);
        mask(1, 1)      = 0;
        mask(end, 1)    = 0;
        mask(1, end)    = 0;
        mask(end, end)  = 0;
        x = x(mask == 1);
        y = y(mask == 1);
        corner_mask = sub2ind(size(mask), y, x);
        %corner_mask = [2:6, 8:42, 44:48];
    else
        corner_mask = 1:(patch_size * patch_size);
    end
    
end