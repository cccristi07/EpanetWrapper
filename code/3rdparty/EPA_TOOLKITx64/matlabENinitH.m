function [ret]=matlabENinitH(flag)

if ~(flag==00 || flag==01 || flag==10 || flag==11)
    error('Error Epanet -> Input Error 251: Illegal parameter code in function call');
    return;
end
ret = calllib('epanet2', 'ENinitH', flag);
% if ret~=0
%     error(['Error Epanet -> ',matlabENgeterror(ret)]);
%     return;
% end
