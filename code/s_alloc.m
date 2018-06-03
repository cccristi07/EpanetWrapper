%%% Copyright (c) 2016, Paul Irofti <paul@irofti.net>
%%% All rights reserved.

function [sn,X] = s_alloc(R, s, osn, strategy)
%% Sensor Allocation routine: placement given residues R and s sensors
% INPUTS:
%   R -- residues matrix
%   s -- number of sensors
%
% OUTPUTS:
%   sn -- sensor nodes
%   osn -- optional: existing sensor nodes (s MUST include these as well)

    if nargin < 3
        osn = [];
    end
    R = double(R);
    % Use the natural base as dictionary and pick the most used atoms
    if nargin < 3 || isempty(osn)
        X = omp(R,eye(size(R,1)),s);
    else
        for i = 1:size(R,2)
            [indv, x] = omp_restart(eye(size(R,1)), R(:,i), s, osn);
            x_omp = zeros(size(R,1),1);
            x_omp(indv(1:s)) = x;
            X(:,i) = x_omp;
        end
    end
    if nargin == 4 && strcmp(strategy, 'block')
        % Find most used atom for each node block
        pitems = size(R,2) / size(R,1);
        block = zeros(size(R,1),1);
        for i = 1:size(R,1)  % for each node
            [rows, ~] = find(X(:,(i-1)*pitems+1:i*pitems));
            tbl = sortrows(tabulate(rows),2);   % find most used atoms
            block(i) = tbl(end,1);
            disp(['Node ' num2str(i) ' most used atom: ' num2str(tbl(end,1))]);
        end
        tbl = sortrows(tabulate(block),2);
        bn = tbl(end-s+1:end,1)';
        %disp(['Sensor nodes per block popularity: ' num2str(bn)]);
        sn = bn;
    else
        % Overall most popular atoms
        [rows, ~] = find(X);
        tbl = sortrows(tabulate(rows),2);   % find most used atoms
        sn = tbl(end-s+1:end,1)';
    end
    
    % Try and use a combination of the least popular with most popular
    %sn = [tbl(1:floor(s/4),1)' tbl(end-(s-floor(s/4)-1):end,1)'];
end