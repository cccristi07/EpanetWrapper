function [qualCode, traceNode] = matlabENgetqualtype()

qualCode=0;
traceNode=0;
[ret, qualCode, traceNode] = calllib('epanet2', 'ENgetqualtype', qualCode, traceNode);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
