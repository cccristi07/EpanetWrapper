function matlabENsettimeparam( paramCode, timeValue)

if ~isnumeric(timeValue)
    timeValue = getenconstant(timeValue);
end
if isnumeric(paramCode)
    if paramCode>3 || paramCode<0
        error('Error Epanet -> Input Error 251: Illegal parameter code in function call');
        return;
    end
    ret = calllib('epanet2', 'ENsettimeparam', paramCode, timeValue);
else
    ret = calllib('epanet2', 'ENsettimeparam', getenconstant(paramCode), timeValue);
end
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
