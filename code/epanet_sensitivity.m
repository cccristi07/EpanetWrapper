%% Gather sensitivity measurements for various junction fault scenarios
clear; close all; fclose all; format compact;
%%-------------------------------------------------------------------------
%% emitter = magnitudine leakage pe retea
emitter=[2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32];     % train overflow
emitter_test=[3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33];% test overflow
network = './3rdparty/networks/hanoi.inp';
report = 'hanoi.txt';       % EPANET log file
ss = 5;                      % number of sensors

% Sensor placement parameters
strategy='overall';  % sensor placement strategy: 'block', 'overall';

% Classification parameters
class_parts = 0;    % Group a few nodes in one class to reduce errors

sparsitythres = 30; % sparsity prior
sqrt_alpha = 4; % weights for label constraint term
sqrt_beta = 2; % weights for classification err term
dictsize = 256; % dictionary size
iterations = 50; % iteration number
iterations4ini = 20; % iteration number for initialization

gendata = 0;                % generate residues
%--------------------------------------------------------------------------
addpath('3rdparty/EPA_TOOLKITx64');
addpath(genpath('3rdparty/LCKSVD'));

%% Get residues via EPANET emulation
if gendata
    [R, H_train] = residues(emitter, network, report);
    [R_test, H_test] = residues(emitter_test, network, report);
    save('residues.mat', 'R', 'H_train', 'R_test', 'H_test');
else
    load('residues.mat', 'R', 'H_train', 'R_test', 'H_test');
end
%% Build labels for subgraphs

if class_parts
    samples = length(emitter);
    H_train = build_labels(parts, samples);

    samples = length(emitter_test);
    H_test = build_labels(parts, samples);
end

% Show classification results at the end
i = 1;
class = zeros(length(ss),1);

for s = ss
disp(['s=' num2str(s)]);

%% Sensors placement
[sensor_nodes,Xs] = s_alloc(R, s, [], strategy);
R0 = R;
R0(setdiff(1:size(R,1),sensor_nodes),:) = 0;
norm(R0-R,'fro')

disp(['Sensor Nodes: ' num2str(sort(sensor_nodes))]);

%% Classification: Learning
training_feats = double(R(sensor_nodes,:));

[Dinit,Tinit,Winit,Q_train] = initialization4LCKSVD(training_feats,H_train,dictsize,iterations4ini,sparsitythres);
[D2,X2,T2,W2] = labelconsistentksvd2(training_feats,Dinit,Q_train,Tinit,H_train,Winit,iterations,sparsitythres,sqrt_alpha,sqrt_beta);
save('dictionarydata2.mat','D2','X2','W2','T2');

% Classification on training data
[prediction,accuracy,err,Xtrain] = ...
    classification(D2, W2, training_feats, H_train, sparsitythres);

%% Classification: Test data
testing_feats = double(R_test(sensor_nodes,:));

% Classification on test data
[prediction2,accuracy2,err2,Xtest] = ...
    classification(D2, W2, testing_feats, H_test, sparsitythres);
%clab = {'errid' 'featid' 'truth' 'pred'};
%disp(array2table(err2, 'VariableNames',clab))
fprintf('Final recognition rate for LC-KSVD2 is : %.03f \n', accuracy2);

end %% ss
class
save residuals_all_info

