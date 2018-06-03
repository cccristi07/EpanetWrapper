function matlabENsetstatusreport(level)

if level>2 || level<0
    error('Error Epanet -> Input Error 251: Illegal parameter code in function call');
    return;
end
ret = calllib('epanet2', 'ENsetstatusreport', level);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
