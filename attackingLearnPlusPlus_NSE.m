n_timestamps = length(data_train);  % total number of time stamps
f_measure = zeros(n_timestamps, net.mclass);
g_mean = zeros(n_timestamps, 1);
recall = zeros(n_timestamps, net.mclass);
precision = zeros(n_timestamps, net.mclass);
err = zeros(n_timestamps, 1);