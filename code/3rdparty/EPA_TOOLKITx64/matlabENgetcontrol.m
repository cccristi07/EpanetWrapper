function [type, lindex, setting, nindex, level] = matlabENgetcontrol( cindex)

count = 0;
[ret, count] = calllib('epanet2', 'ENgetcount', 5, count);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
if cindex>count || cindex<0
    error(['Error Epanet -> Input Error 251: Illegal parameter code in function call']);
    return;
end
type=0;
lindex=0;
setting=0;
nindex=0;
level=0;
[ret, type, lindex, setting, nindex, level] = calllib('epanet2', 'ENgetcontrol', cindex, type, lindex, setting, nindex, level);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
