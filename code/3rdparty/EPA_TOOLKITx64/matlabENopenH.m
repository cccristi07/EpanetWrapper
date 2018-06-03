function [ret]=matlabENopenH()

ret = calllib('epanet2', 'ENopenH');
% if ret~=0
%     error(['Error Epanet -> ',matlabENgeterror(ret)]);
%     return;
% end
