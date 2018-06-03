function ind = matlabENgetnodeindex(indStr)

ind = 0;
[ret, indStr, ind] = calllib('epanet2', 'ENgetnodeindex', indStr, ind);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
