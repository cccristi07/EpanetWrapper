%%% Copyright (c) 2016, Paul Irofti <paul@irofti.net>
%%% All rights reserved.

function H = build_labels(parts, samples)
%% Build node labels for LC-KSVD classification
% INPUTS:
%   parts -- node partitions
%   samples -- samples per node (equals the number of emitters)
%
% OUTPUTS:
%   H -- class labels of input signals
%--------------------------------------------------------------------------
    nodes = max(cell2mat(parts));
    classes = length(parts);
    H = zeros(classes, nodes*samples);
    cmarker = ones(1,samples);
    
    for c = 1:classes
        for n = parts{c}
            H(c, (n-1)*samples + 1:n*samples) = cmarker;
        end
    end
end