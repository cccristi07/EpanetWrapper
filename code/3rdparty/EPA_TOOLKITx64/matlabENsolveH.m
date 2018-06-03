function matlabENsolveH()

ret = calllib('epanet2', 'ENsolveH');
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
