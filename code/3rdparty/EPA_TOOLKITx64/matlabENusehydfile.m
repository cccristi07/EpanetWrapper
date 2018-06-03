function matlabENusehydfile(fName)

[ret,fName] = calllib('epanet2', 'ENusehydfile', fName);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
