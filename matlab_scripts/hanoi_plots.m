clear; clc;


%% load all necesary data
libjson_path = 'C:\Users\Cristian\Desktop\LICENTA\jsonlab-master';
train_data_path = 'C:\Users\Cristian\Desktop\LICENTA\cristian_cazan\EpanetWrapper\ENWrapper\train_set.json';
test_data_path = 'C:\Users\Cristian\Desktop\LICENTA\cristian_cazan\EpanetWrapper\ENWrapper\test_set.json';
test2_data_path = 'C:\Users\Cristian\Desktop\LICENTA\cristian_cazan\EpanetWrapper\ENWrapper\test2_set.json';
addpath(libjson_path);
% train_data = loadjson(train_data_path);
% test_data = loadjson(test_data_path);
% test2_data = loadjson(test2_data_path);
load net_simulations

%% extragem referinta
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
%% reference plot

node_pressurs = ref_pressure(:, 2:end);
plot(node_pressurs)
title('Profil presiuni nominale')
ylabel('Presiunea din noduri mH2O')
xlabel('Timp(*15min)')

min_val = min(node_pressurs,[], 2);
max_val = max(node_pressurs,[], 2);
min_val = min_val';
max_val = max_val';

t = 1:length(min_val);
patch([t t(end:-1:1)], [min_val, max_val(end:-1:1)], 'b')

%% noduri afectate de diferite defecte
%nodul 7 afectat de defectel 7, 17, 25
emitter_node = 25;
emitter_vals = [1, 7, 17, 31];
figure
hold on;
legs = {};
for i = 1:length(emitter_vals)
    
    fault_data = get_emitter_vals(train_data.NODE_VALUES, emitter_vals(i), emitter_node);
    plot(fault_data.EN_PRESSURE(:, 15))
    xlabel('Timp(*15min)')
    ylabel('Presiuni(mH2O)')
    legs{i} = sprintf('Emitter %d in node %d', emitter_vals(i), emitter_node);
end

legend(legs)

emitter_node = 25;
emitter_vals = [35, 40, 50, 100];
figure
hold on;
legs = {};
for i = 1:length(emitter_vals)
    
    fault_data = get_emitter_vals(test2_data.NODE_VALUES, emitter_vals(i), emitter_node);
    plot(fault_data.EN_PRESSURE(:, 15))
    xlabel('Timp(*15min)')
    ylabel('Presiuni(mH2O)')
    legs{i} = sprintf('Emitter %d in node %d', emitter_vals(i), emitter_node);
end

legend(legs)


%% rez absolut
for i = 2:496
    
    vals = train_data.NODE_VALUES{i}.EN_PRESSURE;
    emitter_node = train_data.NODE_VALUES{i}.EMITTER_NODE;
    emitter_vals = train_data.NODE_VALUES{i}.EMITTER_VAL;
    rez = (vals - ref_pressure);
    plot(rez)
    xlabel('Timp(*15min)')
    ylabel('Presiunea din noduri mH2O')
    title(sprintf('emitter in %d vals %d',emitter_node, emitter_vals)) 
%     legend(leg)
    pause(0.1)
    
end

%% rez relativ
for i = 2:496
    
    vals = train_data.NODE_VALUES{i}.EN_PRESSURE;
    emitter_node = train_data.NODE_VALUES{i}.EMITTER_NODE;
    emitter_vals = train_data.NODE_VALUES{i}.EMITTER_VAL;
    rez = (vals - ref_pressure)./ref_pressure;
    plot(rez)
    xlabel('Timp(*15min)')
    ylabel('Presiunea din noduri mH2O')
    title(sprintf('emitter in %d vals %d',emitter_node, emitter_vals)) 
%     legend(leg)
    pause(0.1)
    
end
%% rez standardizat (x - mean(x)) / std(x)
for i = 2:496
    
    vals = test2_data.NODE_VALUES{i}.EN_PRESSURE;
    emitter_node = test2_data.NODE_VALUES{i}.EMITTER_NODE;
    emitter_vals = test2_data.NODE_VALUES{i}.EMITTER_VAL;
    rez = (vals - ref_pressure)./ref_pressure;
    rez = bsxfun(@minus, rez, mean(rez, 1));
    rez = bsxfun(@rdivide, rez, std(rez, 1));
    plot(rez)
    xlabel('Timp(*15min)')
    ylabel('Presiunea din noduri mH2O')
    title(sprintf('emitter in %d vals %d',emitter_node, emitter_vals)) 
%     legend(leg)
    pause(0.1)
    
end

%% rez cu normalizare pe coloane
for i = 2:496
    
    vals = train_data.NODE_VALUES{i}.EN_PRESSURE;
    emitter_node = train_data.NODE_VALUES{i}.EMITTER_NODE;
    emitter_vals = train_data.NODE_VALUES{i}.EMITTER_VAL;
    rez = (vals - ref_pressure);
    rez = normc(rez);
    plot(rez)
    xlabel('Timp(*15min)')
    ylabel('Presiunea din noduri mH2O')
    title(sprintf('emitter in %d vals %d',emitter_node, emitter_vals)) 
%     legend(leg)
    pause(0.1)
    
end

%% medie pe timp a rezlui absolut
for i = 2:496
    
    vals = train_data.NODE_VALUES{i}.EN_PRESSURE;
    emitter_node = train_data.NODE_VALUES{i}.EMITTER_NODE;
    emitter_vals = train_data.NODE_VALUES{i}.EMITTER_VAL;
    rez = (vals - ref_pressure);
    rez = mean(rez(1:35),1);
    plot(rez, 'bx')
    xlabel('Timp(*15min)')
    ylabel('Presiunea din noduri mH2O')
    title(sprintf('emitter in %d vals %d',emitter_node, emitter_vals)) 
%     legend(leg)
    pause(0.1)
    
end
%% medie pe timp a rezlui relativ
for i = 2:496
    
    vals = train_data.NODE_VALUES{i}.EN_PRESSURE;
    emitter_node = train_data.NODE_VALUES{i}.EMITTER_NODE;
    emitter_vals = train_data.NODE_VALUES{i}.EMITTER_VAL;
    rez = ((vals) - (ref_pressure))./(ref_pressure);
    rez = mean(rez(1:35,:));
    plot(rez, 'bx')
    xlabel('Timp(*15min)')
    ylabel('Presiunea din noduri mH2O')
    title(sprintf('emitter in %d vals %d',emitter_node, emitter_vals)) 
%     legend(leg)
    pause(0.1)
    
end

%% emitter de 25 in toate nodurile 

emitters = [15];
for i = 1:length(emitters)
    figure
    hold on
    for node = 2:31
        vals = get_emitter_vals(train_data.NODE_VALUES, emitters(i), node);
        vals = vals.EN_PRESSURE;
        rez = (mean(vals) - mean(ref_pressure))./mean(ref_pressure);
        plot(rez);
    end
end

%% calculam matricea de rezri R pentru dict learning
thresh_hold = 0.05; % pentru rez absolut ales ochiometric
R = [];
%% medie pe timp a rezlui standardizat
for i = 2:496
    
    vals = train_data.NODE_VALUES{i}.EN_PRESSURE;
    emitter_node = train_data.NODE_VALUES{i}.EMITTER_NODE;
    emitter_vals = train_data.NODE_VALUES{i}.EMITTER_VAL;
    rez = (vals - ref_pressure);
    med = mean(rez, 1);
    dev = std(rez, 1);
    rez = bsxfun(@minus, rez, med);
    rez = bsxfun(@rdivide, rez, dev);
    rez = mean(rez(1:35,:));
    plot(rez, 'bx')
    xlabel('Timp(*15min)')
    ylabel('Presiunea din noduri mH2O')
    title(sprintf('emitter in %d vals %d',emitter_node, emitter_vals)) 
%     legend(leg)
    pause(0.1)
    
end
%% medie pe timp a rezlui normalizat pe col
for i = 2:496
    
    vals = train_data.NODE_VALUES{i}.EN_PRESSURE;
    emitter_node = train_data.NODE_VALUES{i}.EMITTER_NODE;
    emitter_vals = train_data.NODE_VALUES{i}.EMITTER_VAL;
    rez = (vals - ref_pressure);
    rez = bsxfun(@minus, rez, mean(rez, 1));
    rez = bsxfun(@rdivide, rez, std(rez, 1));
    rez = mean(rez(1:35,:));
    plot(rez, 'bx')
    xlabel('Timp(*15min)')
    ylabel('Presiunea din noduri mH2O')
    title(sprintf('emitter in %d vals %d',emitter_node, emitter_vals)) 
%     legend(leg)
    pause(0.1)
    
end

%% plots
figure;
for i=1:496
    
    % obtain the residual by substracting the refference
    vals = train_data.NODE_VALUES{i}.EN_PRESSURE;
    vals = mean(vals(1:35,:),1);
    rez = (vals - mean_ref(1:31))./mean_ref(1:31);
    node = train_data.NODE_VALUES{i}.EMITTER_NODE;
    val = train_data.NODE_VALUES{i}.EMITTER_VAL;
    
    plot(rez, 'rx')
    %axis([0, 32, -20, 20])
    title(sprintf('rez for emitter in node %d and val %d', node, val));
    pause(0.25);
    
end

%% plotare features auxiliare EN_DEMAND & EN_VELOCITY
% pentru nodurile 5, 11, 17, 27
% pentru pipes 5, 12, 15, 19
nodes = [5, 11, 17, 27];
pipes = [5, 10, 15, 25];
pipe_legend = {};
for i = 1:length(pipes)
    pipe_legend{i} = sprintf('PIPE %d', pipes(i));
end

figure
plot(ref_velocity(:, pipes))
xlabel('Timp(*15min)')
ylabel('Viteza(m/s)')
legend(pipe_legend)

savefig('C:\Users\Cristian\Desktop\LICENTA\cristian_cazan\LicentaLatex\pics\figs\ref_velocities.fig')
matlab2tikz('C:\Users\Cristian\Desktop\LICENTA\cristian_cazan\LicentaLatex\pics\tikz_files\ref_velocities.tikz')

node_legend = {};
for i = 1:length(nodes)
    node_legend{i} = sprintf('NODE %d', nodes(i));
end

figure
plot(ref_demand(:, nodes))
xlabel('Timp(*15min)')
ylabel('Demand(L/s)')
legend(node_legend)
savefig('C:\Users\Cristian\Desktop\LICENTA\cristian_cazan\LicentaLatex\pics\figs\ref_demands.fig')
matlab2tikz('C:\Users\Cristian\Desktop\LICENTA\cristian_cazan\LicentaLatex\pics\tikz_files\ref_demands.tikz')
%% TODO-uri
% ploturi rez absolut
% pentru acelasi nod i toate fault-urile care il afecteaza
% pentru fiecare nod sa plotez cum e afectat e un anumita magnitudine a
% defectului "plimbata" in fiecare nod
% de plotat toate ploturile pe acelasi grafic - test
% la fel pentru rez relativ
%
% analizat cazul cu velocity vs demand vs pressure - absolut vs relativ
% rez filtrat pe suma de x elemente
% sau rez �ntr-un anumit moment de timp i.e. fara suma
%
% ce se intampla daca variez demand-ul? 
%
%
% TODO metoda pentru modificarea demand-ului pe nod!!!
%
%
% consideram variatii de mag in fault si variatii de demand separat!!!
% for demand in demands:
%   -- primul demand sa fie nominal: referinta
%   -- aleg o magnitudine fixa care sa imi dea rez relevante
% 
%     for nod_afectat_defect in noduri_afectate_defect:
%       2 tipuri de scenarii - variatii de demand & variatii de magnitudine
%       pentru fiecare pot sa arat cum se comp sistemul pressure flow
%       reziduriile pentru fiecare
%       
%   pentru a pune  cod in licenta lstlistings 
%  figure in tikz
% 
%
%

% P = rand(31, 100);
% S = zeros(31, 100);
% S = [zeros(31, 1) -double(P < 0.33) + double(P > 0.67)];
% D = dnom + ddelta * S(:, i); % sa fie si zerouri pe acolo
% 
% save fault_sign
    
  