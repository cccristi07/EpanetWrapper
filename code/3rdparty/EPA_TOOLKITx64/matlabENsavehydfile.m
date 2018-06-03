function matlabENsavehydfile(fName)

[ret,fName] = calllib('epanet2', 'ENsavehydfile', fName);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
