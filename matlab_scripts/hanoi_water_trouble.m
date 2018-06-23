clear; clc; close all;

%% incarcam seturile de date
load('net_simulations')

%% parametrii pentru dictionar
dict_size = 256;
s = 4; % sparsitate
alpha = 4; % penalizare clasificare
beta = 16; % label consistent penalizare
init_method = 1; % train small dictionaries for each class at init
nsensors = 2:10;

emitter=[2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32];     % train overflow
emitter_test=[3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33];% test overflow

sensors = {
 % nodes with sensors
[10, 11, 25, 27],
[10, 11, 16, 25, 27, 28],
[9, 10, 11, 12, 16, 20, 25, 27, 28, 29],
[11, 15,21,28],
[12,13,16,21,26,28],
[6,12,13,14,15,16,21,26,27,28]
};
%% construiesc matricile de reziduuri
H_train = [];
H_test = [];
H_test2 = [];
R_train = [];
R_test = [];
R_test2 = [];

nv_train = train_data.NODE_VALUES;
nv_test = test_data.NODE_VALUES;
nv_test2 = test2_data.NODE_VALUES;



parts = {[1 2 3 4 16 17 18], [5 6 7 8 13 14 15], [9 10 11 12],  ...
   [19 20 21 22], [23 24 25 26 27 28 29 30 31]};

p_ref = nv_train{1}.EN_PRESSURE;
%% date de antrenare
for node = 1:31
    for emitter_val = 1:31
        if mod(emitter_val, 2) == 1 && nv_train{1}.EMITTER_VAL ~= 0
            sim_data = get_emitter_vals(nv_train, emitter_val, node);
            h = zeros(31, 1);
            h(node) = 1;
            % matricea de labels
            H_train = [H_train, h];
            sim_pressure = sim_data.EN_PRESSURE;
            % absolute residual
            rez = mean(sim_pressure(2:25,:)) - mean(p_ref(2:25, :));
            rez = rez(:);
            R_train = [R_train, rez];
        end
    end
end

%% date de test
for node = 1:31
    for emitter_val = 1:31
        if mod(emitter_val, 2) == 0
            sim_data = get_emitter_vals(nv_test, emitter_val, node);
            h = zeros(31, 1);
            h(node) = 1;
            % matricea de labels
            H_test = [H_test, h];
            sim_pressure = sim_data.EN_PRESSURE;
            % absolute residual
            rez = mean(sim_pressure(2:25,:)) - mean(p_ref(2:25, :));
            rez = rez(:);
            R_test = [R_test, rez];
        end
    end
end
%% date test strong
for i = 1:length(nv_test)
    
    if nv_test{i}.EMITTER_VAL == 0
        continue 
    end
    
    sim_data = nv_test{i};
    node = sim_data.EMITTER_NODE;
    
    h = zeros(31, 1);
    h(node) = 1;
    % matricea de labels
    H_test2 = [H_test2, h];
    sim_pressure = sim_data.EN_PRESSURE;
    % absolute residual
    rez = mean(sim_pressure(2:25,:)) - mean(p_ref(2:25, :));
    rez = rez(:);
    R_test2 = [R_test2, rez];
end
results = zeros(2,length(nsensors),2);
H_test = H_test2;
R_test = R_test2;
%% Save junction labels
H_train_junc = H_train;
H_test_junc = H_test;
%% Build labels for subgraphs    
samples = 16;
H_train_parts = build_labels(parts, samples);

samples = 15;
H_test_parts = build_labels(parts, samples);

for isensors = 1:length(nsensors)
    sensor_nodes = sensors{isensors};
    disp(['Sensor Nodes: ' num2str(sort(sensor_nodes))]);

    for class_parts = [0]
        disp(['class_parts=' num2str(class_parts)]);

        %% Build labels for subgraphs    
        if class_parts
            H_train = H_train_parts;
            H_test = H_test_parts;
        else
            H_train = H_train_junc;
            H_test = H_test_junc;
        end

        %% Classification: Learning
        Y_train = double(R_train(sensor_nodes,:));

        % Compute labels
        c = size(H_train,1);        % number of classes
        nc = floor(dict_size/c);            % evenly divide atoms per classes
        nr = nc*c;                  % total number of atoms (nr <= n)
        Q_train = zeros(nr, size(Y_train,2));
        jj = 0;
        for i = 1 : c                 % allocate atoms for each signal
          jc = find(H_train(i,:)==1); % indices of signals from class i
          Q_train(jj+1:jj+nc,jc) = 1;
          jj = jj + nc;
        end

        % Perform DL
        [W1, D1] = clas_discrim_dl(Y_train, H_train, ...
            nr, s, alpha, init_method);
        [W2, D2, ~] = clas_labelcon_dl(Y_train, H_train, Q_train, ...
            nr, s, alpha, beta, init_method);

        %% Classification: Test data
        Y_test = double(R_test(sensor_nodes,:));

        accuracy1 = classification(Y_test, H_test, D1, W1, s);
        accuracy2 = classification(Y_test, H_test, D2, W2, s);

        results(class_parts + 1, isensors, 1) = accuracy1;
        results(class_parts + 1, isensors, 2) = accuracy2;
        fprintf('discriminative DL %.03f and LC-KSVD %.03f \n', ...
            accuracy1, accuracy2);
    end %% clas_parts
end %% ss

%% Plot
figure(1);
plot(nsensors,results(1, :,1));
hold on;
plot(nsensors,results(1, :,2));
legend({'discriminative DL', 'LC-DL'});
title(['Junctions (c=' num2str(size(H_train_junc,1)) ')']);
hold off;

figure(2);
plot(nsensors,results(2, :,1));
hold on;
plot(nsensors,results(2, :,2));
legend({'discriminative DL', 'LC-DL'});
title(['Partitions (c=' num2str(size(H_train_parts,1)) ')']);
hold off;

