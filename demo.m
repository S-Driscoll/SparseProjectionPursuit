% Script for demonstrating sparse projection pursuit analysis.
% Requires mat file Salmon.mat and SPPA.m in directory or PATH
% Data are NMR of Salmon blood plasma samples (5 classes, 15 in each except
% 1)

 load Salmon.mat 
% Rows of X contain spectra, class contains the sample classes, chemshift contains ppm axis

[T,V,Vars]=SPPA(X,'dim',2,'nvars',5,'meth','mul')

% Plot results color coding to class membership
figure
colorvec='rgbyk';
for i=1:length(T)
    scatter(T(i,1),T(i,2),'MarkerFaceColor',colorvec(class(i)),'MarkerEdgeColor','k','LineWidth',1.5)
    hold on
end
xlabel('Score 1')
ylabel('Score 2')
set(gca,'LineWidth',2,'FontSize',14,'FontWeight','bold')