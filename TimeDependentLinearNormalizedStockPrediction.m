% This script file sets up a Linear Neural Network in order to predict the
% future values of the S&P 500 index as a function of time and its past
% values. The daily values of the stock market index are normalized through
% logarithmization. The time variable is normalized within the [1/N ... 1]
% interval.

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

% Plot the original time series.
figure('Name','Original S&P 500 Time Series');
plot(t,X,'-.','LineWidth',1.8);
ylabel('Adjusted Closing Value');
xlabel('Time');
grid on

% Normalize variables under investigation.
X_norm = log(X);
t_norm = t / N;

% Plot normalized time series.
figure('Name','Normalized S&P 500 Time Series');
plot(t_norm,X_norm,'-.','LineWidth',1.8);
ylabel('Adjusted Closing Value');
xlabel('Time');
grid on

% Set the time window of past values to be utilized.
time_window = 10;

% Generate the appropriate sequences of past time series  
% (normalized) data for the given time window.
[Xps_norm,Tps] = generate_past_sequences(X_norm,time_window);

% Set the percentage of available data instances to be used for training.
training_percentage = 0.90;

% Set training and testing (normalized) data instances. 
[Xps_norm_train,Xps_norm_test,Tps_train,Tps_test] = generate_training_testing_data(Xps_norm,Tps,training_percentage);

% -------------------------------------------------------------------------
% TRAINING MODE:
% -------------------------------------------------------------------------
% Set training data patterns and corresponding targets (normalized).
Pnorm = [Xps_norm_train(:,2:end),Tps_train];
Tnorm = Xps_norm_train(:,1);
% Transposition of training patterns and corresponding targets (normalized).
Pnorm = Pnorm';
Tnorm = Tnorm';

% Get the corresponding minmax matrix for the input vectors. The number of
% rows for the R matrix will be equal to the number of columns of Pnorm
% matrix. Each row of matrix R contains a pair of minumum and maximum
% values for the correspondin column of matrix Pnorm. In other words, each
% row element of matrix R provides the limits for the corresponding input
% feature.
R = minmax(Pnorm);

% Set the linear neural network for the normalized version of the input
% variables.
net  = newlin(R,1,0,0.1);
% Initialize network object.
net = init(net);
net.inputWeights{1,1}.learnParam.lr = 10^(-9);
net.biases{1}.learnParam.lr = 10^(-9);
net.trainParam.epochs = 1000;
net.trainParam.goal = 0.0;
net.trainFcn = 'trainb';
% Train network object.
net = train(net,Pnorm,Tnorm);

% Get network predictions on training data.
Yps_norm_train = sim(net,Pnorm);
% Compute the correspodning RMSE value.
RMSE_norm_train = sqrt(mean((Yps_norm_train-Tnorm).^2));

% -------------------------------------------------------------------------
% TESTING MODE:
% -------------------------------------------------------------------------
% Set testing data patterns and corresponding targets (normalized).
Pnorm = [Xps_norm_test(:,2:end),Tps_test];
Tnorm = Xps_norm_test(:,1);
% Transposition of training patterns and corresponding targets (normalized).
Pnorm = Pnorm';
Tnorm = Tnorm';

% Get network predictions on normalized testing data.
Yps_norm_test = sim(net,Pnorm);
% Compute the correspodning RMSE value for the normalized testing data.
RMSE_norm_test = sqrt(mean((Yps_norm_test-Tnorm).^2));
% Plot corresponding fitting performance.
figure_name = 'S&P 500 Normalized Values';
plot_fitting_performance(figure_name,Xps_norm_train(:,1),Tps_train,Yps_norm_train,Xps_norm_test(:,1),Tps_test,Yps_norm_test)

% Output training and testing performance metrics in terms of RMSE.
fprintf('RMSE TRAINING: %f\n',RMSE_norm_train);
fprintf('RMSE TESTING: %f\n',RMSE_norm_test);

% Get the internal network parameter values after training. For the case of
% a linear network these parameters correspond to the weight vector and the
% associated bias term. The weight vector will be stored in variable W
% which is a [1 x (time_window+1)] row vector accounting for the past
% values of the time series which are included in the prediction model plus
% the time parameter. The bias term will be stored in variable B.
W = net.IW{1,1};
B = net.b{1};