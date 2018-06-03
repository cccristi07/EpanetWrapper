%%% Copyright (c) 2016, Paul Irofti <paul@irofti.net>
%%% All rights reserved.

function [R, L] = residues(emitter, network, report)
%% Walk each junction within the network and fetch the pressures
% INPUTS:
%   emitter -- vector of liters per second leaks
%   network -- EPANET network input file (inp)
%   report -- optional log
%
% OUTPUTS:
%   R -- residues: ordered blocks by junction ID, each block containing
%                  residue vectors correspoding to each emitter entry
%--------------------------------------------------------------------------
    loadlibrary('epanet2', 'epanet2');

    if nargin < 3
        report = 'epanet.log';
    end

    matlabENopen(network, report, '');

    % The number of junctions in a network equals the number of nodes minus
    % the number of tanks and reservoirs.
    junctions =  matlabENgetcount('EN_NODECOUNT') - ...
                 matlabENgetcount('EN_TANKCOUNT');

    % residual for each junction fault scenario
    R = [];
    % Label matrix: each row represents a class (leak in our case)
    % and each column corresponds to the residue measurement 
    % with a single non-zero entry marking the leaky junction
    L = zeros(junctions,length(emitter)*junctions);

    matlabENopenH();

    % Get nominal pressure in each junction
    P = all_junctions_get_pressure(junctions);

    lstart = 1;
    lend = 0;
    for m=1:junctions
        fprintf('\njunction %d:', mod(m,10));
        for e=1:length(emitter)
            fprintf(' %d', emitter(e));
            %% simulam faptul ca nodul M este în defect
            matlabENsetnodevalue(m, 'EN_EMITTER',emitter(e));
            %%
            % Get junction pressures with leak in node m 
            P_leakm = all_junctions_get_pressure(junctions); % val la regim stationar
            lend = lend + size(P_leakm,2);
            R = [R (P-P_leakm)]; % reziduu = dif dintre nominal si masurat 

            matlabENsetnodevalue(m, 'EN_EMITTER',0);
        end
        L(m, lstart:lend) = 1;
        lstart = lend + 1;
    end

    matlabENcloseH();
    matlabENclose();
    unloadlibrary('epanet2');
end