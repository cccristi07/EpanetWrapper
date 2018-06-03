function ind = matlabENgetlinkindex(indStr)

ind = 0;
[ret, indStr, ind] = calllib('epanet2', 'ENgetlinkindex', indStr,ind);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
