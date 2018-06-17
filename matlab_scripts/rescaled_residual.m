function [ res ] = rescaled_residual( ref, measured )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    

    abs_res = measured - ref;
    min_res = min(abs_res, [], 1);
    max_res = max(abs_res, [], 1);
    
    res = bsxfun(@minus, abs_res, min_res);
    res = bsxfun(@rdivide, res, max_res - min_res);
end

