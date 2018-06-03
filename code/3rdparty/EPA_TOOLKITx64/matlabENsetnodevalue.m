function [ret]=matlabENsetnodevalue( lindex, code, value)

    count = 0;
    [ret, count] = calllib('epanet2', 'ENgetcount', 0, count);
    if ret~=0
        error(['Error Epanet -> ',matlabENgeterror(ret)]);
        return;
    end
    if lindex>count || lindex<1
        error('Error Epanet -> Input Error 251: Illegal parameter code in function call');
        return;
    end
    if isnumeric(code)
        if code>8 || code<0
            error(['Error Epanet -> Input Error 251: Illegal parameter code in function call']);
            return;
        end
        ret = calllib('epanet2', 'ENsetnodevalue', lindex, code, value);
    else
        ret = calllib('epanet2', 'ENsetnodevalue', lindex, getenconstant(code), value);
    end
    if ret~=0
        error(['Error Epanet -> ',matlabENgeterror(ret)]);
        return;
    end
end