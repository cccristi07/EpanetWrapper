function err = matlabENgeterror(errcode)

str='                                                                                                     ';
[ret,err] = calllib('epanet2', 'ENgeterror', errcode, str, 99);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
