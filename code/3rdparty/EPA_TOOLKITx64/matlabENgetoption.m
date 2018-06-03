function value = matlabENgetoption(code)

value=0;
if isnumeric(code)
    if code>4 || code<0
        error('Error Epanet -> Input Error 251: Illegal parameter code in function call');
        return;
    end
    [ret, value] = calllib('epanet2', 'ENgetoption', code,value);
else
    [ret, value] = calllib('epanet2', 'ENgetoption', getenconstant(code),value);
end
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
