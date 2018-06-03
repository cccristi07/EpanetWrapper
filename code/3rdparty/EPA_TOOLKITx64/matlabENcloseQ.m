function matlabENcloseQ()

ret = calllib('epanet2', 'ENcloseQ');
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
