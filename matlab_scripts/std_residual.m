function [ rez ] = std_residual( ref, measured )
% se calculeaza reziduul standardizat (res - mean(res)) / std(res)


    rez = (measured - ref);
    rez = bsxfun(@minus, rez, mean(rez, 1));
    rez = bsxfun(@rdivide, rez, std(rez, 1));


end

