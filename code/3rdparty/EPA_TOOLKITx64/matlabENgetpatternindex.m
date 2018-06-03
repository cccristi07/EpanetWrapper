function ind = matlabENgetpatternindex(indStr)

ind=0;
[ret, indStr, ind] = calllib('epanet2', 'ENgetpatternindex', indStr,ind);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
