function matlabENsetpatternvalue( lindex, period, value)
%% functie care seteaza profilul de utilizare pentru un nod
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
ret = calllib('epanet2', 'ENsetpatternvalue', lindex, period, value);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
