function ver = matlabENgetversion()

ver=0;
[ret, ver] = calllib('epanet2', 'ENgetversion', ver);
if ret~=0
    error(['Error Epanet -> ',matlabENgeterror(ret)]);
    return;
end
