function matlabENsaveH()

ret = calllib('epanet2', 'ENsaveH');
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
