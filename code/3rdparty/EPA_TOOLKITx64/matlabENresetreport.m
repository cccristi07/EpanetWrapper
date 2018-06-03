function matlabENresetreport()

ret = calllib('epanet2', 'ENresetreport');
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
