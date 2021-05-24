% Load all data

topics = csvread('topics.csv');

fid = fopen('runs.csv');
runs = textscan(fid,'%s','delimiter',',');
runs = runs{1,1};

ndcg = csvread('ndcg.csv', 0, 0);

% Compute the mean performance of each run across topics
% This is the MAP for each run, if we loaded the AP data
m = mean(ndcg);

% sort in descending order of mean score
[m_sorted, idx] = sort(m, 'descend');

% re-order runs by descending mean of the measure
% needed to have a more nice looking box plot
ndcg = ndcg(:, idx);
runs = runs(idx);

figure

% show the box plot
boxchart(ndcg)

hold on

% plot the mean on top of the box plot
plot (m_sorted, ':o', "MarkerSize", 10, "LineWidth", 2)

% adjust tick labels on x-axis, y-axis range, and font size
ax = gca;
ax.FontSize = 16;
ax.XTickLabel = [];

xlabel("Runs (decreasing order of mean nDCG@5)")
ylabel("Performance (nDCG@5)")