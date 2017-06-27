function [Xps,Tps] = generate_past_sequences(X,time_window)

% This function generates past sequence data for a given time series and
% time_window in order to capture dependencies on past values of the index
% under investigation. Xps will be an [time_window+1 x N-time_window] 
% matrix storing column-wise the corresponding time series data for the 
% previous time instances. That is, the k-th column of matrix Xps [Xps(:,k)] 
% will be storing the index values for the time instances t' such that:
% time_window+k <= t' <= N - k + 1. Tps is the corresponding vector of
% normalized time 

N = length(X);
Xps = zeros(time_window+1,N-time_window);
for k = 0:1:time_window
    Xps(k+1,:) = X(time_window+1-k:1:N-k);
end;
Xps = Xps';
Nps = N - time_window;
Tps = [1:1:Nps]';
Tps = Tps ./ Nps;


end

