function plot_fitting_performance(figure_name,x_train,t_train,y_train,x_test,t_test,y_test)

% This function plots the fitting performance of the utilized neural
% network for both training and testing modes of operation.

figure('Name',figure_name);
hold on
plot(t_train,x_train,'-.b','LineWidth',1.8);
plot(t_test,x_test,'-.k','LineWidth',1.8);
plot(t_train,y_train,'-.g','LineWidth',1.8);
plot(t_test,y_test,'-.r','LineWidth',1.8);
hold off
grid on
xlabel('Time');
ylabel('Index Values');

end

