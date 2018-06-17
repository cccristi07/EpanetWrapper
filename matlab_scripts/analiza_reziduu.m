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

%% reziduu relativ - medie pe timp
% aleg un emitter standard de 15 pentru a crea matricea de Reziduuri R care
% are dimensiunile (n_faults, n_nodes) n_faults reprezinta emitter_ul
% simulat în fiecare nod al retelei
emitter_val = 23;
node_vals = train_data.NODE_VALUES;
pipe_vals = train_data.LINK_VALUES;
figure(1)
hold on
for i = 2:31
    
    sim_vals = get_emitter_vals(node_vals, emitter_val, i);
    measured_pressure = sim_vals.EN_PRESSURE;
    scores = abs_residual(ref_pressure, measured_pressure);
    rez = mean(scores(1:35, :));

    plot(rez, 'x');
    title(sprintf('emitter in node %d', i))
    pause(0.25)
end

    