function ENT_MAP = extract_entmap(data, answer)
ENT_MAP = {};

col = size(data, 2);
lcol = size(answer, 2);

f_ent = zeros(col, 1);
fl_ent = zeros(col, lcol);
ff_ent = zeros(col, col);

for k = 1:col
    f_ent(k, 1) = p_entropy(data(:, k));
    for l = 1:lcol
        fl_ent(k, l) = p_entropy([data(:, k), answer(:, l)]);
    end
    for l = 1:col
        ff_ent(k, l) = p_entropy([data(:, k), data(:, l)]);
    end
end
    
ENT_MAP{1,1} = f_ent;
ENT_MAP{1,2} = fl_ent;
ENT_MAP{1,3} = ff_ent;
end

function [res] = p_entropy( vector )

[uidx,~,single] = unique( vector, 'rows' );
count = zeros(size(uidx,1),1);
for k=1:size(vector,1)
    count( single(k), 1 ) = count( single(k), 1 ) + 1;
end
res = -( (count/size(vector,1))'*log2( (count/size(vector,1)) ) );
end
