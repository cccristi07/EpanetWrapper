function matlabENsetreport(command)

[ret,command] = calllib('epanet2', 'ENsetreport', command);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
