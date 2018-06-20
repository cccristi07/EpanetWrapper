clear; clc; close all;

%% incarcam setul de date
load net_simulations
%% extragere referinta
ref_pressure = train_data.NODE_VALUES{1}.EN_PRESSURE; % nu exista nici o scurgere
ref_demand = train_data.NODE_VALUES{1}.EN_DEMAND; %
ref_velocity = train_data.LINK_VALUES{1}.EN_VELOCITY;

node_values = train_data.NODE_VALUES;
link_values = train_data.LINK_VALUES;

ref_pressure_mean = mean(ref_pressure(1:35,:),1);
ref_demand_mean = mean(ref_demand(1:35, :),1);
ref_velocity_mean = mean(ref_velocity(1:35, :),1);

leg = {};

for i = 1:31
    leg{i} = sprintf('NODE %d', i);
end
%% reziduu 
%% reziduu relativ - medie pe timp
% aleg un emitter standard de 15 pentru a crea matricea de Reziduuri R care
% are dimensiunile (n_faults, n_nodes) n_faults reprezinta emitter_ul
% simulat în fiecare nod al retelei
emitter_val = 25;
emitter_nodes = [11, 17, 21, 27];
node_vals = train_data.NODE_VALUES;
pipe_vals = train_data.LINK_VALUES;
for i = 1:length(emitter_nodes)
    
    sim_vals = get_emitter_vals(node_vals, emitter_val, emitter_nodes(i));
    measured_pressure = sim_vals.EN_PRESSURE;
    scores = abs_residual(ref_pressure, measured_pressure);
    rez = normc(scores);
    rez = mean(rez(5:20, :));
    
    figure
    plot(rez, 'bx');
    xlabel('Nod')
    savefig(sprintf('atem_res_emitter%d_mag%d', emitter_nodes(i),emitter_val));
    matlab2tikz(sprintf('atem_res_emitter%d_mag%d.tikz', emitter_nodes(i),emitter_val))
end
%% reziduu timp normalizare coloane
% nodurile [11, 17, 21]
% emitter  [ 5, 15 ,29]
nodes = [27];
emitter = 29;
nodes2plot = [5, 11, 15, 17, 21, 27];
l = {};
for i = 1:length(nodes2plot)
    l{i} = sprintf('Node %d', nodes2plot(i));
end
for i = 1:length(nodes)
    sim_vals = get_emitter_vals(node_vals, 29, nodes(i));
    measured_pressure = sim_vals.EN_PRESSURE(:, nodes2plot);
    scores = abs_residual(ref_pressure(:, nodes2plot), measured_pressure);
%     rez = mean(scores(5:20, :));
    rez = normc(scores);
    figure
    plot(rez);
    xlabel('Timp(*15min)')
    ylabel('Reziduu presiune(mH2O)')
    legend(l, 'Location', 'southeast')
    savefig(sprintf('time_res_emitter%d_mag29', nodes(i)));
    matlab2tikz(sprintf('time_res_emitter%d_mag29.tikz', nodes(i)))

%     legend(leg)    
end

