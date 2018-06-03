function [from,to] = matlabENgetlinknodes(linkInd)

count=0;
[ret, count] = calllib('epanet2', 'ENgetcount', 2, count);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
if linkInd>count || linkInd<1
    error('Error Epanet -> Input Error 251: Illegal parameter code in function call');
    return;
end
from=0;
to=0;
[ret, from, to] = calllib('epanet2', 'ENgetlinknodes', linkInd, from, to);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
