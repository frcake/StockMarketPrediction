% This script file sets up a Multi-Layer Feed-Forward Neural Network 
% in order to predict the future values of the S&P 500 index as a function 
% of its past values. The daily values of the stock market index
% are not normalized through logarithmization. 

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
C= zeros(0,101);
% Plot the original time series.
figure('Name','Original S&P 500 Time Series');
plot(t,X,'-.','LineWidth',1.8);
ylabel('Adjusted Closing Value');
xlabel('Time');
grid on
%for i=1:100
% Set the time window of past values to be utilized.
time_window = 7;

% Generate the appropriate sequences of past time series  
% (un-normalized) data for the given time window.
[Xps,Tps] = generate_past_sequences(X,time_window);

% Set the percentage of available data instances to be used for training.
training_percentage = 0.70;

validation_percentage = 0.20;

% Set training and testing (un-normalized) data instances. 
[Xps_train,Xps_test,Tps_train,Tps_test] = generate_training_testing_data(Xps,Tps,training_percentage,validation_percentage);

% -------------------------------------------------------------------------
% TRAINING MODE:
% -------------------------------------------------------------------------
% Set training data patterns and corresponding targets (un-normalized).
P = Xps_train(:,2:end);
T = Xps_train(:,1);
% Transposition of training patterns and corresponding targets (un-normalized).
P = P';
T = T';

% Set the neural network for the un-normalized version of input variables.
net = newff(P,T,[8 4 4 1],{'tansig' 'tansig' 'tansig' 'purelin'});
% Initialize network object.
init(net);
% Set internal network parameters.
net.trainParam.epochs = 1000;
net.trainParam.showCommandLine = 1;
net.trainParam.show = 1;
net.trainParam.goal = 0.000001;
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
% Transposition of training patterns and corresponding targets (un-normalized).
P = P';
T = T';

% Get network predictions on un-normalized testing data.
Yps_test = sim(net,P);
Z = minus(T,Yps_test);
%C(i) = sum(Z);
%fprintf('SUB: %f\n',sum(C));
%end
% Compute the correspodning RMSE value for the un-normalized testing data.
RMSE_test = sqrt(mean((Yps_test-T).^2));
% Plot corresponding fitting performance.
figure_name = 'S&P 500 UnNormalized Values';
plot_fitting_performance(figure_name,Xps_train(:,1),Tps_train,Yps_train,Xps_test(:,1),Tps_test,Yps_test)

% Output training and testing performance metrics in terms of RMSE.
fprintf('RMSE TRAINING: %f\n',RMSE_train);
fprintf('RMSE TESTING: %f\n',RMSE_test);