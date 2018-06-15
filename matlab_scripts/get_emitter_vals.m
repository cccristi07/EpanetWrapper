function [ ret_data ] = get_emitter_vals( data, emitter_val, emitter_node)
% Function that returns the values of a certain emitter node simulation
% it's based upon the structure of the simulation

    ret_data = {};
   for i = 1:length(data)
       
       if data{i}.EMITTER_VAL == emitter_val && data{i}.EMITTER_NODE == emitter_node
           ret_data = data{i};
       end
   end
   

end

