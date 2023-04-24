function [pre,post] = pmlbayes_matlab( train, answer, test )
% Multi Label Naive Bayes
lcol = size( answer, 2 );
pre = zeros( size(test,1), lcol );
post = zeros( size(test,1), lcol );

for k=1:lcol
    model = fitcnb( train, answer(:,k),  'DistributionNames', 'mvmn' );

    [pre(:,k),t] = predict( model, test  );
    t(isnan(t(:,end)),end) = 0;
    pre(isnan(pre(:,k)),k) = 0;

    post(:,k) = t(:,end);
end
end