function [ret,tStep] = matlabENrunH()

tStep = 0;
[ret, tStep] = calllib('epanet2', 'ENrunH', tStep);
% if ret~=0
%     error(['Error Epanet -> ',matlabENgeterror(ret)]);
%     return;
% end
