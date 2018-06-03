function matlabENsetlinkvalue( lindex, code, value)

count = 0;
[ret, count] = calllib('epanet2', 'ENgetcount', 2, count);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
if lindex>count || lindex<1
    error('Error Epanet -> Input Error 251: Illegal parameter code in function call');
    return;
end
if isnumeric(code)
    if code>12 || code<0
        error('Error Epanet -> Input Error 251: Illegal parameter code in function call');
        return;
    end
    ret = calllib('epanet2', 'ENsetlinkvalue', lindex, code, value);
else
    ret = calllib('epanet2', 'ENsetlinkvalue', lindex, getenconstant(code), value);
end
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
