function value = matlabENgetpatternvalue( lindex, pattern)

count = 0;
[ret, count] = calllib('epanet2', 'ENgetcount', 3, count);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
if lindex>count || lindex<1
    error('Error Epanet -> Input Error 251: Illegal parameter code in function call');
    return;
end
value=0;
[ret, value] = calllib('epanet2', 'ENgetpatternvalue', lindex, pattern, value);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
