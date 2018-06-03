function [ret]=matlabENclose()

ret = calllib('epanet2', 'ENclose');
% if ret~=0
%     error(['Error Epanet -> ',matlabENgeterror(ret)]);
%     return;
% end
