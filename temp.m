clear;

dataName = 'scene';

data_path = './Dataset/';
out_path = './result/';
ext = '.mat';

load(sprintf('%s%s%s',data_path,dataName,ext), "data_dis", "answer", "sim_seq")

data = data_dis;

exp_iter = 10;

warning( 'off' )
pop_size = 20;
max_ffcs = 50;
f_size = 50;
for k=1:exp_iter
    k
    train_data = data( sim_seq(:,k), : );
    train_answer = answer( sim_seq(:,k), : );
    test_data = data( ~sim_seq(:,k), :);
    scores = extract_entmap(train_data, train_answer);

    perf_proposed(k) = struct( 'rloss', [0 0 0], 'onerr', [0 0 0], 'mlacc', [0 0 0], 'mlcov', [0 0 0] );
    
    [stats_proposed_rloss{k,1}, stats_proposed_rloss{k,2}] = proposed_prob( train_data, train_answer, pop_size, max_ffcs, f_size, 'rloss', scores);
    [pre,~] = pmlbayes_matlab( train_data(:,stats_proposed_rloss{k,1}(end,:)==1), ...
            train_answer, test_data(:, stats_proposed_rloss{k,1}(end,:)==1 ) );
    perf_proposed(k).rloss = rloss( answer( ~sim_seq(:,k), : ), pre );
    
    [stats_proposed_onerr{k,1},stats_proposed_onerr{k,2}] = proposed_prob( train_data, train_answer, pop_size, max_ffcs, f_size, 'onerr', scores);
    [~,post] = pmlbayes_matlab( train_data(:,stats_proposed_onerr{k,1}(end,:)==1), ...
        train_answer, test_data(:, stats_proposed_onerr{k,1}(end,:)==1 ) );
    perf_proposed(k).onerr = onerr( answer( ~sim_seq(:,k), : ), post );
    
    [stats_proposed_mlacc{k,1}, stats_proposed_mlacc{k,2}] = proposed_prob( train_data, train_answer, pop_size, max_ffcs, f_size, 'mlacc', scores);
    [pre,~] = pmlbayes_matlab( train_data(:,stats_proposed_mlacc{k,1}(end,:)==1), ...
        train_answer, test_data(:, stats_proposed_mlacc{k,1}(end,:)==1 ) );
    perf_proposed(k).mlacc = mlacc( answer( ~sim_seq(:,k), : ), pre );
    
    [stats_proposed_mlcov{k,1},stats_proposed_mlcov{k,2}] = proposed_prob( train_data, train_answer, pop_size, max_ffcs, f_size, 'mlcov', scores);
    [pre,~] = pmlbayes_matlab( train_data(:,stats_proposed_mlcov{k,1}(end,:)==1), ...
        train_answer, test_data(:, stats_proposed_mlcov{k,1}(end,:)==1 ) );
    perf_proposed(k).mlcov = mlcov( answer( ~sim_seq(:,k), : ), pre );

end

save(sprintf('%s%s%s',out_path,dataName,ext))

