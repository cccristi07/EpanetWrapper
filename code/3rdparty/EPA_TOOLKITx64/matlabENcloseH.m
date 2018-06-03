function [ret]=matlabENcloseH()

ret = calllib('epanet2', 'ENcloseH');
% if ret~=0
%     error(['Error Epanet -> ',matlabENgeterror(ret)]);
%     return;
% end
