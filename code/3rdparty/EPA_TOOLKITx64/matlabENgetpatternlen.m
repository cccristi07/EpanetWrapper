function len = matlabENgetpatternlen(index)

count = 0;
[ret, count] = calllib('epanet2', 'ENgetcount', 3, count);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
if index>count || index<1
    error('Error Epanet -> Input Error 251: Illegal parameter code in function call');
    return;
end
len=0;
[ret, len] = calllib('epanet2', 'ENgetpatternlen', index, len);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
