function grad_map = apply_grad_filters(img, filters)

    grad_map = zeros(size(img, 1), size(img, 2), length(filters));
    for i = 1:length(filters)
        grad_map(:, :, i) = conv2(img, filters{i}, 'same');
    end

end
