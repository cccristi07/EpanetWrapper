function matlabENsetpattern( lindex, factors, nFactors)
%% functie pentru setarea de 
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
if (isnumeric(factors)&& length(factors) == nFactors)
    [ret, factors] = calllib('epanet2', 'ENsetpattern', lindex, factors, nFactors);
    if ret~=0
        error(['Error Epanet -> ',matlabENgeterror(ret)]);
        return;
    end
else
    error('Error Epanet -> Input Error 251: Illegal parameter code in function call');
    return;
end
    
