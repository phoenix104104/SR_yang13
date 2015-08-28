function num = compute_fcollect_feature_size(feature_list)
    
    num = 0;
    for i = 1:length(feature_list)
        num = num + size(feature_list{i}, 1);
    end

end
