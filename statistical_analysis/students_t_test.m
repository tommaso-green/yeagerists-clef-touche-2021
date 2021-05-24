% Load all data

topics = csvread('topics.csv');

fid = fopen('runs.csv');
runs = textscan(fid,'%s','delimiter',',');
runs = runs{1,1};

ndcg = csvread('ndcg.csv', 0, 0);

[n_runs, ~] = size(runs);

% Compute the number of possible pairs of runs to compare
m = (n_runs * (n_runs - 1)) / 2;

% Find the coorect alpha value according to Bonferroni correction,
% because we want to keep under control Family-Wise Error Rate
common_alpha = 0.05;            % Significance level alpha is usually 5%
alpha = common_alpha / m;
fprintf("Currently using alpha: %f\n\n", alpha);

% r1 and r2 are the indexes of the two runs to compare
for r1 = 1:n_runs
    fprintf("Running Student's t test on run '%s':\n", runs{r1});
    for r2 = 1:n_runs
        if r1 ~= r2
            % If two runs have exactly identical scores => ttest returns NaN
            [h, p] = ttest(ndcg(:, r1), ndcg(:, r2), "Alpha", alpha);
            if h == 1
                fprintf("Run '%s' vs '%s': null-hyp rejected -> %d | p-value -> %f\n", runs{r1}, runs{r2}, h, p);
            end
        end
    end
    fprintf("\n");
end