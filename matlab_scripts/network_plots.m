clear; clc;


%% loading datasets
DATA_PATH = 'net_simulations';
load(DATA_PATH)

%% plotting every node
node = 15;
ref = train_data.NODE_VALUES{1}.EN_PRESSURE;
figure
semilogy(ref(:, node));
hold on
for i = 3:10:400
    ref = train_data.NODE_VALUES{i}.EN_PRESSURE;
    semilogy(ref(:, node));
end
