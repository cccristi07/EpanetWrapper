function [ret]=matlabENopen(inFile, repFile, outFile)

[ret, inFile, repFile, outFile] = calllib('epanet2', 'ENopen', inFile, repFile, outFile);
% if ret~=0
%     error(['Error Epanet -> ',matlabENgeterror(ret)]);
%     return;
% end
