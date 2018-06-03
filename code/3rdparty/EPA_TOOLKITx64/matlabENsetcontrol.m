function matlabENsetcontrol( cindex, type, lindex, setting, nindex, level)

count = 0;
[ret, count] = calllib('epanet2', 'ENgetcount', 5, count);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
if cindex>count || cindex<1
    error('Error Epanet -> Input Error 251: Illegal parameter code in function call');
    return;
end
if isnumeric(type)
    if type>3 || type<0
        error('Error Epanet -> Input Error 251: Illegal parameter code in function call');
        return;
    end
    ret = calllib('epanet2', 'ENgetcontrol', cindex, type, lindex, setting, nindex, level);
else
    ret = calllib('epanet2', 'ENgetcontrol', cindex, getenconstant(type), lindex, setting, nindex, level);
end
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
