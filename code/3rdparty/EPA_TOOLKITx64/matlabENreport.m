function matlabENreport()

ret = calllib('epanet2', 'ENreport');
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end