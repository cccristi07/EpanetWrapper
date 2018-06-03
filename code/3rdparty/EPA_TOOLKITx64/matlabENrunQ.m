function tStep = matlabENrunQ()

tStep=0;
[ret, tStep] = calllib('epanet2', 'ENrunQ', tStep);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
