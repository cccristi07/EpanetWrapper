function [ data ] = getENdata( n, fetch_data)
%% Walk each junction within the network and fetch required data
% INPUTS:
%   n -- total number of junctions
%
% OUTPUTS:
%   p -- pressure values
%   ptrans -- transitive pressure values
    tstep=1;
    itrans=1;   % transitive values counter
    data = containers.Map;
    % 10 = Don't save data and reinit links
    matlabENinitH(10);

    % 15 minutes is the observed length of time until the next hydraulic
	% event occurs that ENnextH() returns.
    % We stop when this value is no longer positive.
    while tstep>0
        matlabENrunH();

        for i=1:n
            ptrans(i,itrans)=matlabENgetnodevalue(i,'EN_PRESSURE');
        end
        tstep=matlabENnextH();
        itrans = itrans + 1;
    end
    p = ptrans(:,end);  % final pressure values
end
