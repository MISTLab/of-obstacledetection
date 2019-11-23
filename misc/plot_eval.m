clear all
close all
clc

folder = '/data/test/model/evaluation/';

fnames = {'predicted_and_real_labels' 
          'predicted_and_real_steerings'
          'constant_regression' 
          'random_classification'
          'random_regression'
          'test_classification'
          'test_regression'};

% Read data from Json files
for f = 1:numel(fnames)
    fsearch_name = folder + string(fnames{f}) + '*';
    cur_fnames = dir(char(fsearch_name));
    
    for i = 1:numel(cur_fnames)
        cur_fname = strcat(cur_fnames(i).folder, '/', cur_fnames(i).name);
        res.(fnames{f})(i) = jsondecode(fileread(cur_fname));
    end
end

j = 0:i-1;

% Plot Eval
figure(1)

%Steering
subplot(3,3,1)
hold on
plot(j, [res.(fnames{7}).evas], '-r');
plot(j, [res.(fnames{5}).evas], '-g');
plot(j, [res.(fnames{3}).evas], '-b');
title('Steer.: EVAs');
xlabel('Epoch');

subplot(3,3,2)
hold on
plot(j, [res.(fnames{7}).rmse], '-r');
plot(j, [res.(fnames{5}).rmse], '-g');
plot(j, [res.(fnames{3}).rmse], '-b');
title('Steer.: RMSE');
xlabel('Epoch');

subplot(3,3,7)
hold on
plot(j, mean([res.(fnames{7}).highest_errors]), '-r');
plot(j, mean([res.(fnames{5}).highest_errors]), '-g');
plot(j, mean([res.(fnames{3}).highest_errors]), '-b');
legend('test','random','constant')
title('Steer.: Highest Errors');
xlabel('Epoch');

subplot(3,3,4)
hold on
dat_s = [res.(fnames{2}).pred_steerings] - [res.(fnames{2}).real_steerings];
plot(j, mean(dat_s), '-r');
plot(j, std(dat_s), '-g');
for k = 1:size(dat_s,2)
    plot(dat_s(:,k), '-');
end
title('Steer.: av. Diff. & STD');
xlabel('Epoch');

% Collision
subplot(3,3,3)
hold on
plot(j, [res.(fnames{6}).ave_accuracy], '-m');
plot(j, [res.(fnames{4}).ave_accuracy], '-c');
title('Coll.: av. Accuracy');
xlabel('Epoch');

subplot(3,3,6)
hold on
plot(j, [res.(fnames{6}).precision], '-m');
plot(j, [res.(fnames{4}).precision], '-c');
title('Coll.: Precision');
xlabel('Epoch');

subplot(3,3,9)
hold on
plot(j, [res.(fnames{6}).f_score], '-m');
plot(j, [res.(fnames{4}).f_score], '-c');
title('Coll.: f-Score');
xlabel('Epoch');

subplot(3,3,5)
hold on
dat_c = [res.(fnames{1}).pred_probabilities] - [res.(fnames{1}).real_labels];
plot(j, mean(dat_c), '-c');
plot(j, std(dat_c), '-m');
title('Coll.: av. Diff. & STD');
xlabel('Epoch');

subplot(3,3,8)
hold on
plot(j, mean([res.(fnames{6}).highest_errors]), '-m');
plot(j, mean([res.(fnames{4}).highest_errors]), '-c');
legend('test','random')
title('Coll.: Highest Errors');
xlabel('Epoch');