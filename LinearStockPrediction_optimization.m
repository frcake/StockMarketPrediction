% This script file sets up a Linear Neural Network in order to predict the
% future values of the S&P 500 index as a function of its past values.

clc
clear all

% Load daily time series data from file: stock_data/SP500.mat
load('stock_data\SP500.mat');

% SP500: is a structure variable containing daily values for quantities that
% specifically relate to the S&P 500 stock market index for the time period  
% between 1/1/1950 and 1/1/2015, that is a total amount of 16355 entries.

% The individual variables stored within the structure variable include the 
% following fields:
% (i)   AdjClose: row vector storing the adjusted closing value per day.
% (ii)  Date: cell array storing row-wise the corresponding dates.
% (iii) High: row vector storing the highest index value  per day.
% (iv)  Low: row vector storing the highest index value per day.
% (v)   Close: row vector storing the closing index value per day.
% (vi)  Open: row vector storing the opening index value per day.
% (vi)  Volume: row vector storing the daily volume of transactions.

% Set the variable X under investigation to be the adjusted closing value
% of the index.
X = SP500.AdjClose;

% Get the number of observations.
N = length(X);

% Set X observations in ascending time order.
X = X(N:-1:1);

% Set the vector of corresponding time instances.
t = [1:1:N];

% Normalize the time vector.
t_norm = t ./ N;

% Plot the original time series.
figure('Name','Original S&P 500 Time Series');
plot(t_norm,X,'-.','LineWidth',1.8);
ylabel('Adjusted Closing Value');
xlabel('Time');
grid on
for i = 1:4
% Set the time window of past values to be utilized.
time_window = i;

% Generate the appropriate sequences of past time series  
% (un-normalized) data for the given time window.
[Xps,Tps] = generate_past_sequences(X,time_window);

% Set the percentage of available data instances to be used for training.
training_percentage = 0.70;
validation_percentage = 0.20;
%test_percentage = 0.10;

% Set training and testing (un-normalized) data instances. 
[Xps_train,Xps_val,Xps_test,Tps_train,Tps_val,Tps_test] = generate_training_testing_data_mine(Xps,Tps,training_percentage,validation_percentage);

% -------------------------------------------------------------------------
% TRAINING MODE:
% -------------------------------------------------------------------------
% Set training data patterns and corresponding targets (un-normalized).
P = Xps_train(:,2:end);
T = Xps_train(:,1);
% Transposition of training patterns and corresponding targets (un-normalized).
P = P';
T = T';

% Get the corresponding minmax matrix for the input vectors. The number of
% rows for the R matrix will be equal to the number of columns of Pnorm
% matrix. Each row of matrix R contains a pair of minumum and maximum
% values for the correspondin column of matrix Pnorm. In other words, each
% row element of matrix R provides the limits for the corresponding input
% feature.
R = minmax(P);

% Set the linear neural network for the normalized version of the input
% variables.
net  = newlin(R,1,0,0.1);
% Initialize network object.
net = init(net);
net.inputWeights{1,1}.learnParam.lr = 10^(-11);
net.biases{1}.learnParam.lr = 10^(-11);
net.trainParam.epochs = 1000;
net.trainParam.goal = 0.0;
net.trainFcn = 'trainb';
% Train network object.
net = train(net,P,T);

% Get network predictions on training data.
Yps_train = sim(net,P);
% Compute the correspodning RMSE value.
RMSE_train = sqrt(mean((Yps_train-T).^2));

% -------------------------------------------------------------------------
% TESTING MODE:
% -------------------------------------------------------------------------
% Set testing data patterns and corresponding targets (un-normalized).
P = Xps_test(:,2:end);
T = Xps_test(:,1);
Pval = Xps_val(:,2:end);
Tval = Xps_val(:,1);
% Transposition of training patterns and corresponding targets (un-normalized).
P = P';
T = T';
Pval = Pval';
Tval = Tval';

% Get network predictions on un-normalized testing data.
Yps_val = sim(net,Pval);
Yps_test = sim(net,P);
%C = minus(T,Yps_test);
%fprintf('c %f\n',C);
%fprintf('SUM: %f\n',sum(C));
%end
% Compute the correspodning RMSE value for the un-normalized testing data.
RMSE_test = sqrt(mean((Yps_test-T).^2));
RMSE_val = sqrt(mean((Yps_val-Tval).^2));
% Plot corresponding fitting performance.
figure_name = 'S&P 500 UnNormalized Values';
%plot_fitting_performance(figure_name,Xps_train(:,1),Tps_train,Yps_train,Xps_val(:,1),Tps_val,Yps_val,Xps_test(:,1),Tps_test,Yps_test)

% Output training and testing performance metrics in terms of RMSE.
fprintf('RMSE TRAINING: %f\n',RMSE_train);
fprintf('RMSE TESTING: %f\n',RMSE_test);
fprintf('RMSE VAL: %f\n',RMSE_val);
Gtrain(i) = RMSE_train;
Gval(i) = RMSE_val;
Gtest(i) = RMSE_test;
end
plot(1:length(Gtrain),Gtrain,1:length(Gtrain),Gval,1:length(Gtrain),Gtest);
%plot([Gtrain Gval Gtest]);
% Get the internal network parameter values after training. For the case of
% a linear network these parameters correspond to the weight vector and the
% associated bias term. The weight vector will be stored in variable W
% which is a [1 x (time_window+1)] row vector accounting for the past
% values of the time series which are included in the prediction model plus
% the time parameter. The bias term will be stored in variable B.
W = net.IW{1,1};
B = net.b{1};