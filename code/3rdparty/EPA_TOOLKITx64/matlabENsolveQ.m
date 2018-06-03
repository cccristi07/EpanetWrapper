function matlabENsolveQ()

ret = calllib('epanet2', 'ENsolveQ');
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
