function [count,ret] = matlabENgetcount(code)

count = 0;
if isnumeric(code)
    if code>5 || code<0
        error('Error Epanet -> Input Error 251: Illegal parameter code in function call');
        return;
    end
    [ret, count] = calllib('epanet2', 'ENgetcount', code, count);
else
    [ret, count] = calllib('epanet2', 'ENgetcount', getenconstant(code), count);
end
% if ret~=0
%     error(['Error Epanet -> ',matlabENgeterror(ret)]);
%     return;
% end
