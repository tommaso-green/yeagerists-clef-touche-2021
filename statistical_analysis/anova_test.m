%{
% Load all data (for 2020 results)
topics = csvread('topics.csv');

fid = fopen('runs_selected.csv');
all_runs = textscan(fid,'%s','delimiter',',');
all_runs = all_runs{1,1};

all_ndcg = csvread('ndcg_selected.csv', 0, 0);
%}

% Load all data (for 2021 results)
topics = csvread('topics_2021.csv');

fid = fopen('runs_2021.csv');
all_runs = textscan(fid,'%s','delimiter',',');
all_runs = all_runs{1,1};

all_ndcg = csvread('nDCG_per_topic_2021.csv', 0, 0);

%{
% Consider only the 5 runs submitted on Tira
% See on wandb.ai -> tag "tira_submission"
top_runs_indexes = [67 86 26 279 242 127 2 30 3 103];
top_runs = all_runs(top_runs_indexes);
top_ndcg = all_ndcg(:, top_runs_indexes);
%}

% the mean for each run across the topics
% Note that if the measure is AP (Average Precision), 
% this is exactly MAP (Mean Average Precision) for each run
m = mean(all_ndcg);

% sort in descending order of mean score
[m_sorted, idx] = sort(m, 'descend');

% re-order runs by descending mean of the measure
all_ndcg = all_ndcg(:, idx);
all_runs = all_runs(idx);

% perform the ANOVA
%[~, tbl, stats] = anova1(all_ndcg, all_runs, 'off');
[~, tbl, stats] = anova2(all_ndcg, 1, 'off');

% display the ANOVA table
tbl

% the significance level
alpha = 0.05;

c = multcompare(stats, 'Alpha', alpha, 'Ctype', 'hsd');

% display the multiple comparisons
%c

