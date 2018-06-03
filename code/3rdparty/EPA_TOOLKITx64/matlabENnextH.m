function [tStep,ret] = matlabENnextH()

tStep = 0;
[ret,tStep] = calllib('epanet2', 'ENnextH', tStep);
% if ret~=0
%     error(['Error Epanet -> ',matlabENgeterror(ret)]);
%     return;
% end
