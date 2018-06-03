function idStr = matlabENgetlinkid(index)

count = 0;
[ret, count] = calllib('epanet2', 'ENgetcount', 2, count);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
if index>count || index<1
    error('Error Epanet -> Input Error 251: Illegal parameter code in function call');
    return;
end
idStr='                     ';
[ret,idStr] = calllib('epanet2', 'ENgetlinkid', index, idStr);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
