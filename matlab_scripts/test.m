clear; clc;

%% fault detection via set covering problem
% care sunt cele mai importante noduri pentru detectia fault-ului

M = double(rand(32) > 0.9);
M
alpha = binvar(size(M,2), 1);

constraints = [];
size(M(1,:))
size(alpha)
% 
% for i = 1:size(M,1)
%     
%     constraints = [constraints, ...
%         M(i,:) *alpha >= 1];
% end

objective = sum(alpha);

optimize(M*alpha >=1, sum(alpha))
alpha = value(alpha);
sum(alpha) % numarul de noduri care algoritmul de optimizare numerica a selectat
sum(M(alpha == 1, :),2) >= 1 % verif daca se indeplineste conditia de set covering