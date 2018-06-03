function matlabENsaveinpfile(fName)

[ret,fName] = calllib('epanet2', 'ENsaveinpfile', fName);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
