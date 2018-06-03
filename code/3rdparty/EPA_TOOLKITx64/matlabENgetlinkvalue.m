function value = matlabENgetlinkvalue( lindex, code)

count = 0;
[ret, count] = calllib('epanet2', 'ENgetcount', 2, count);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
if lindex>count || lindex<1
    disp('Error Epanet -> Input Error 251: Illegal parameter code in function call');
    return;
end
value=0;
if isnumeric(code)
    if code>13 || code<0
        error('Error Epanet -> Input Error 251: Illegal parameter code in function call');
        return;
    end
    [ret, value] = calllib('epanet2', 'ENgetlinkvalue', lindex, code, value);
else
    [ret, value] = calllib('epanet2', 'ENgetlinkvalue', lindex, getenconstant(code), value);
end
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end

