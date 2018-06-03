function matlabENsetoption(code, value)

if isnumeric(code)
    if code>4 || code<0
        error('Error Epanet -> Input Error 251: Illegal parameter code in function call');
        return;
    end
    ret = calllib('epanet2', 'ENsetoption', code, value);
else
    ret = calllib('epanet2', 'ENsetoption', getenconstant(code), value);
end
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
