function [typeCode,ret] = matlabENgetnodetype(index)

count = 0;
[ret, count] = calllib('epanet2', 'ENgetcount', 0, count);
% if ret~=0
%     error(['Error Epanet -> ',matlabENgeterror(ret)]);
%     return;
% end
if index>count || index<1
    error('Error Epanet -> Input Error 251: Illegal parameter code in function call');
    return;
end
typeCode=0;
[ret, typeCode] = calllib('epanet2', 'ENgetnodetype', index, typeCode);
% if ret~=0
%     disp(['Error Epanet -> ',matlabENgeterror(ret)]);
%     return;
% end
