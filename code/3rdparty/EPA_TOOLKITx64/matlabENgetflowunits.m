function code = matlabENgetflowunits()

code=0;
[ret, code] = calllib('epanet2', 'ENgetflowunits', code);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
