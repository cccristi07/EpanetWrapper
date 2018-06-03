function matlabENsetqualtype(qualCode, chemname, chemunits, traceNode)

if isnumeric(qualCode)
    if qualCode>3 || qualCode<0
        error('Error Epanet -> Input Error 251: Illegal parameter code in function call');
        return;
    end
    [ret, chemname, chemunits, traceNode] = calllib('epanet2', 'ENsetqualtype', qualCode, chemname, chemunits, traceNode);
else
    [ret, chemname, chemunits, traceNode] = calllib('epanet2', 'ENsetqualtype', getenconstant(qualCode), chemname, chemunits, traceNode);
end
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
