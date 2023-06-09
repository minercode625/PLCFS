function [res] = onerr( answer, predict )

[trow tcol] = size(answer);
[prow pcol] = size(predict);

if (trow ~= prow) || (tcol ~= pcol)
    error( 'The size of answer and predict must be same size' );
end

res = 0;
for k=1:trow
    [val idx] = max( predict(k,:) );
    if answer(k,idx) ~= 1
        res = res + 1;
    end
end
res = res / trow;
