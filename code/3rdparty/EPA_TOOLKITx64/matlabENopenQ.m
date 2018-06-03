function matlabENopenQ()

ret = calllib('epanet2', 'ENopenQ');
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
