%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% step_04_plot_eval_coll.m
% 
% Fourth step of workflow: plot evaluation data from previous step.
% 
% Written by Moritz Sperling
% 
% Licensed under the MIT License (see LICENSE for details)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all

addpath(pwd);
addpath(strcat(pwd, '/util'));

global idx full_res res windowSize thresh fnames;
thresh = 0.5;
windowSize = 3; 
idx = 1;

folder = '/data/experiment_rootdir/eval/evaluation/';

fnames = {'predicted_and_real_labels' 
          'random_classification'
          'test_classification'};
      

% Read data from Json files
for f = 1:numel(fnames)
    fsearch_name = folder + string(fnames{f}) + '*';
    cur_fnames = dir(char(fsearch_name));
    
    for i = 1:numel(cur_fnames)
        cur_fname = strcat(cur_fnames(i).folder, '/', cur_fnames(i).name);
        full_res.(fnames{f})(i) = jsondecode(fileread(cur_fname));
        if f == 1
            res{i} = full_res.(fnames{1})(i);
        end
    end
end

% read folder names
fsearch_name = strcat(folder, 'predicted_and_real_labels*');
folnames = dir(char(fsearch_name));
[lbls, pred, falseclea, falsecoll, subdirs, conf, conff, clearp, collis, X, Z, j] = calc_data();

% build gui
ss = get(0,'ScreenSize');
ui = uifigure;
ui.Name = 'Current Experiment';
ui.Position = [0 0 ss(3)*0.4 ss(4)*0.6];
ax1 = uiaxes(ui, 'Position', [10 220 500 200]);
ax2 = uipanel(ui, 'Position', [300 20 220 180]);
uilabel(ui, ...
        'Position', [10 20 80 22],...
        'Text', 'Select Folder: ');
dd = uidropdown(ui, ...
    'Position', [100 20 150 22], ...
    'Items', {subdirs{X}}, ...
    'Value', subdirs{1}, ...
    'ValueChangedFcn', @(dd,event) plot_selection(dd, ax1, ax2, subdirs));
uilabel(ui, ...
        'Position', [10 50 80 22],...
        'Text', 'Select Epoch: ');
de = uidropdown(ui, ...
    'Position', [100 50 150 22], ...
    'Items', {folnames.name}, ...
    'ValueChangedFcn', @(de,event) change_epoch(de, dd, ax1, ax2, subdirs, folnames));

% open gui
plot_all(subdirs, X, Z);
plot_epoch(lbls, falseclea, falsecoll, subdirs, conf, conff, clearp, collis, j);
plot_selection(dd, ax1, ax2, subdirs);

function [lbls, pred, falseclea, falsecoll, subdirs, conf, conff, clearp, collis, X, Z, j] = calc_data()
    global idx res windowSize thresh;
    
    % data and labels
    lbls = [res{idx}.real_labels];
    pred = [res{idx}.pred_probabilities];
    
    % smooth data
    f = (1/windowSize)*ones(1,windowSize);
    predf = filter(f, 1, pred);

    % get overall detected clear path and detected collisions (true 0, true 1)
    conf = confusionmat(lbls, double(pred > thresh));
    conff = confusionmat(lbls, double(predf > thresh));
    perc0 = conff(1) / sum(lbls == 0);
    perc1 = conff(4) / sum(lbls == 1);

    % Get filenames from predictions and select unique directories
    subdirs = {};
    ftmp = res{idx}.filenames;
    for i = 1:size(ftmp,1)
        imgpath = regexp(ftmp{i},'/','split');
        if (contains(imgpath{1}, 'collision') || contains(imgpath{1}, 'GOPR'))
            subdirs{end + 1} = imgpath{1};
        end
    end
    [~,X,Z] = unique(subdirs','stable');

    % iterate trough unique dirs and split data into sets
    for j = 1:size(X,1)
        idxs = Z == j;
        set_l = lbls(idxs);
        set_p = pred(idxs);
        set_pt = double(set_p > thresh);
        num_clr = sum(set_l == 0);
        num_col = sum(set_l == 1);

        % confusion matrices for each set
        cm = confusionmat(set_l, set_pt);
        if (numel(cm) > 1)
            clearp(j,1) = cm(1) / num_clr * 100;
            clearp(j,2) = cm(3) / num_clr * 100;
            collis(j,1) = cm(4) / num_col * 100;
            collis(j,2) = cm(2) / num_col * 100;
        else
            clearp(j,1) = cm / num_clr * 100;
            clearp(j,2) = 100 - clearp(j,1);
            collis(j,1) = 0;
            collis(j,2) = 0;
        end
    end

    falsecoll = (pred-lbls) .* double((pred-lbls) >= 0);
    falseclea = (pred-lbls) .* double((pred-lbls) < 0);
    
    
    collis_p = collis(:,1);
    collis_p(isnan(collis_p)) = 50;
    clearp_p = clearp(:,1);
    clearp_p(isnan(clearp_p)) = 50;
    clearp_p(isinf(clearp_p)) = 50;
    
    disp(strcat("Epoch ", num2str(idx-1), ": "))
    disp(strcat("Overall Detected Collisions: ", num2str(perc1*100), "%; ", ...
                "Falsely Predicted Collisions: ", num2str((1-perc0)*100), "%"))
    disp(strcat("Set STDs - Collision: ", num2str(std(collis_p)), "%; ", ...
                "Clear Path: ", num2str(std(clearp_p)), "%"))
end

function change_epoch(de, dd, ax1, ax2, subdirs, fnames)
    global idx;
    idx = find(contains({fnames.name}, de.Value));
    plot_selection(dd, ax1, ax2, subdirs);
    [lbls, ~, falseclea, falsecoll, subdirs, conf, conff, clearp, collis, X, Z, j] = calc_data();
    plot_epoch(lbls, falseclea, falsecoll, subdirs, conf, conff, clearp, collis, j);
    plot_all(subdirs, X, Z);
end

function plot_all(subdirs, X, Z)
    global full_res idx fnames thresh windowSize;

    % difference between label and predictions
    dat_c = [full_res.(fnames{1}).pred_probabilities] - [full_res.(fnames{1}).real_labels];

    % get detected clear path and detected collisions (true 0, true 1)
    lbls = [full_res.(fnames{1}).real_labels];
    data = [full_res.(fnames{1}).pred_probabilities];
    
    % smooth data
    f = (1/windowSize)*ones(1,windowSize);
    datf = filter(f, 1, data);
    datt = double(datf > thresh);
    
    for i = 1:size(lbls,2)
        conf = confusionmat(lbls(:,i), datt(:,i));
        perc0(i) = conf(3) / sum(lbls(:,1) == 0);
        perc1(i) = conf(4) / sum(lbls(:,1) == 1);
    end

    % find best model
    [m1,i1] = max(perc1-perc0);
    disp(strcat("Model with best Collision Detection: ", num2str(i1-1), ....
                "; Score:  ",num2str(m1*100), "/100",...
                "; Collisions: ", num2str(perc1(i1)*100), "%", ...
                "; Path Clear: ", num2str((1-perc0(i1))*100), "%"));
    
    % calculate percentages per experiment    
    for i = 1:size(lbls,2) 
        for j = 1:size(X,1)
            idxs = Z == j;
            run = lbls(idxs,i);
            runp = datt(idxs,i);
            realcolli = run == 1;
            realclear = run == 0;
            detecollperc(j,i) = sum(runp(realcolli)) / sum(realcolli);
            detecleaperc(j,i) = 1 - sum(runp(realclear)) / sum(realclear);
        end
    end

    % find best experiments
    detecollperc(isnan(detecollperc)) = -Inf;
    curcp = [detecollperc(:,idx) - detecleaperc(:,idx) [1:numel(detecollperc(:,idx))]'];
    sortb = sortrows(curcp, 1, 'descend');
    best5 = sortb(1:5,:);
    detecollperc(isinf(detecollperc)) = NaN;
    bestdirs = {subdirs{X(best5(:,2))}};
    disp(strcat("Best sets: ", bestdirs{1}, " (", num2str(best5(1,1)*100), "); ", ...
                               bestdirs{2}, " (", num2str(best5(2,1)*100), "); ", ...
                               bestdirs{3}, " (", num2str(best5(3,1)*100), "); ", ...
                               bestdirs{4}, " (", num2str(best5(4,1)*100), "); ", ...
                               bestdirs{5}, " (", num2str(best5(5,1)*100), "%)"))

    % find worst experiments
    curcp(isinf(curcp)) = NaN;
    sortw = sortrows(curcp, 1, 'ascend');
    worst5 = sortw(1:5,:);
    worstdirs = {subdirs{X(worst5(:,2))}};
    disp(strcat("Worst sets: ", worstdirs{1}, " (", num2str(worst5(1,1)*100), "); ", ...
                                worstdirs{2}, " (", num2str(worst5(2,1)*100), "); ", ...
                                worstdirs{3}, " (", num2str(worst5(3,1)*100), "); ", ...
                                worstdirs{4}, " (", num2str(worst5(4,1)*100), "); ", ...
                                worstdirs{5}, " (", num2str(worst5(5,1)*100), ")"))

    % Plot Eval
    j = 1:i;
    f2 = figure(2);
    f2.Name = 'All Epochs';
    f2.MenuBar = 'none';
    f2.ToolBar = 'none';
    f2.Units = 'normalized';
    f2.Position = [0 0.65 1 0.35];

    % Collision
    subplot(1,4,1)
    plot(j, [full_res.(fnames{3}).ave_accuracy], '-r');
    hold on
    plot(j, [full_res.(fnames{2}).ave_accuracy], '-c');
    plot(j, [full_res.(fnames{3}).precision], '-g');
    plot(j, [full_res.(fnames{2}).precision], '-m');
    plot(j, [full_res.(fnames{3}).f_score], '-b');
    plot(j, [full_res.(fnames{2}).f_score], '-y');
    plot([idx idx], [0 1], '--k');
    hold off
    title('av. Accuracy, Precision & F1-Score');
    xlabel('Epoch');
    ylim([0,1])
    xlim([-0.5, numel(j) + 0.5])
    l1 = legend('Accuracy model','Accuracy random', ...
                'Precision model','Precision random', ...
                'F1-Score model','F1-Score random', 'Selected Epoch');
    l1.Units = 'pixels';
    l1.Position = [20 220 140 75];
    l1.Title.String = 'Figure 1';
    
    subplot(1,4,2)
    plot(j, mean(dat_c), '-c');
    hold on
    plot(j, std(dat_c), '-m');
    plot(j, ones(size(perc0)) - perc0)
    plot(j, perc1)
    plot(j, perc1 - perc0, '-g')
    plot([idx idx], [0 1], '--k');
    hold off
    l2 = legend('Mean difference', 'Std.', 'True Clear', 'True Collision', 'Score', 'Selected Epoch');
    l2.Units = 'pixels';
    l2.Position = [20 35 140 75];
    l2.Title.String = 'Figure 2';
    title(['Classification with thresh.: ', num2str(thresh)]);
    ylabel('Percentage');
    xlabel('Epoch');
    ylim([0,1])
    xlim([-0.5, numel(j) + 0.5])
    
    ax2 = subplot(1,4,3);
    imagesc(dat_c)
    hold on
    if size(X,1) < 20
        for i = 1:size(X,1)
            plot([0, size(lbls,2)+2], [X(i), X(i)], ':g', 'Linewidth', 2);
        end
    end
    caxis([-1,1])
    colormap(ax2, redblue(256))
    colorbar
    title('Diff. between Label and Prediction')
    ylabel('Image');
    xlabel('Epoch');

    ax1 = subplot(1,4,4);
    imagesc(detecollperc + detecleaperc - 1)
    hold on
    caxis([0,1])
    colorbar
    colormap(ax1, 'parula')
    title('Experiment Q Scores')
    ylabel('Experiment');
    xlabel('Epoch');
end

function plot_epoch(lbls, falseclea, falsecoll, subdirs, conf, conff, clearp, collis, j) 

    % prep figure
    f = figure(3);
    f.MenuBar = 'none';
    f.ToolBar = 'none';
    f.Name = 'Current Epoch';
    f.Units = 'normalized';
    f.Position = [0.4 0 0.6 0.6];
    
    subplot(2,3,[1 2 3])
    hold on
    bar(lbls, 'g');
    bar(falseclea, 'r')
    bar(falsecoll, 'b')
    title('Difference between Ground Truth and Predictions')
    xlabel('Image [#]')
    ylabel('Difference')
    [~,x1,~] = unique(sortrows(subdirs'),'stable');
    if (numel(x1) < 100)
        for k = 2:numel(x1)
            plot([x1(k), x1(k)], [-1, 1], '--m')
        end
    end
    legend('Ground Truth', 'False Clear', 'False Collision', 'End of Set', 'Location', 'northeastoutside')

    subplot(2,3,4)
    confusionchart(conf, {'Clear', 'Collision'});
    title('Overall Confusion Matrix')
    ylabel('Ground Truth')
    xlabel('Predictions')

    subplot(2,3,5)
    confusionchart(conff, {'Clear', 'Collision'});
    title('Overall Smoothed Confusion Matrix')
    ylabel('Ground Truth')
    xlabel('Predictions')

    subplot(2,3,6)
    bar([clearp collis], 'stacked', 'EdgeColor', 'none')
    title('Collision / Clear Detection Rate')
    ylim([0 200])
    xlim([0.5 j+0.5])
    ylabel('Percentage [%]')
    xlabel('Set [#]') 
end

function plot_selection(dd, ax1, ax2, subdirs)
    global idx res windowSize thresh;
    
    % data and labels
    lbls = [res{idx}.real_labels];
    pred = [res{idx}.pred_probabilities];

    % get indives of set
    fol = dd.Value;
    idxs = find(contains(sortrows(subdirs'),fol));
    
    % get values of set
    lset_l = double(lbls(idxs));
    lset_p = pred(idxs);
    lset_pt = double(lset_p > thresh);
    
    % smooth data
    f = (1/windowSize)*ones(1,windowSize);
    lset_pf = filter(f, 1, lset_p);
    lset_pft = double(lset_pf > thresh);
    
    % confusion matrices for each set
    cm = confusionmat(lset_l, lset_pft);
    
    % plot set
    bar(ax1, lset_l, 'g', 'EdgeColor', 'none');
    hold(ax1, 'on');
    bar(ax1, lset_p, 'r', 'EdgeColor', 'none');
    plot(ax1, lset_pf, '-k', 'LineWidth', 2);
    p = plot(ax1, lset_pft, '--k', 'LineWidth', 2);
    hold(ax1, 'off');
    ax1.Title.String = 'Predictions and Labels';
    ax1.YLim = [0 1];
    ax1.XLabel.String = 'Image #';
    ax1.YLabel.String = 'Predicted Value';
    
    if (numel(cm) > 1)
        cc = confusionchart(ax2, cm, {'Clear', 'Collision'});
        cc.title('Confusion Matrix')
        cc.ylabel('Ground Truth')
        cc.xlabel('Predictions')
    end
end