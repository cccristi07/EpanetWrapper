function matlabENwriteline(strLine)

[ret,strLine] = calllib('epanet2', 'ENwriteline', strLine);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
