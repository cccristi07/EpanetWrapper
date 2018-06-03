function tStep = matlabENstepQ()

tStep=0;
ret = calllib('epanet2', 'ENstepQ', tStep);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
