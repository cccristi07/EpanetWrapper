function tStep = matlabENnextQ()

tStep=0;
ret = calllib('epanet2', 'ENnextQ', tStep);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
