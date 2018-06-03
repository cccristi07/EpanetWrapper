function timeValue = matlabENgettimeparam(paramCode)
% Linea afegida per Gerard
code=paramCode;
% Fins aqui
timeValue=0;
if isnumeric(code)
    if code>9 || code<0
        error('Error Epanet -> Input Error 251: Illegal parameter code in function call');
        return;
    end
    [ret, timeValue] = calllib('epanet2', 'ENgettimeparam', paramCode, timeValue);
else
    [ret, timeValue] = calllib('epanet2', 'ENgettimeparam', getenconstant(paramCode), timeValue);
end
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
