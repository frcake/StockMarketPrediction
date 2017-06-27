function [Xps_train,Xps_test,Tps_train,Tps_test] = generate_training_testing_data(Xps,Tps,training_percentage)

% This function generates training and testing patterns given the past time
% series sequences and the corresponding time instances stored in matrices
% Xps and Tps. The fraction of data instances to be utilized as training
% patterns is determined by the input parameter trainining_percentage.

Nps = length(Tps);
cutoff = round(training_percentage * Nps);
Tps_train = Tps(1:cutoff);
Tps_test = Tps(cutoff+1:Nps);
Xps_train = Xps(1:cutoff,:);
Xps_test = Xps(cutoff+1:Nps,:);

end