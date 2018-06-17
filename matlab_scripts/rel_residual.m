function [ res ] = rel_residual( ref, measured )
% functie care calculeaza reziduul relativ intre doua valori


    res = (ref - measured)./(ref);

end

