function matlabENinitQ(flag)

if ~(flag==0 || flag==1)
    error('Error Epanet -> Input Error 251: Illegal parameter code in function call');
    return;
end
ret = calllib('epanet2', 'ENinitQ', flag);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
