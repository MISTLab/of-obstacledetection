%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% step_06_plot_comparison.m
% 
% Sixth step of workflow: comparison of data from previous step of two models.
% 
% Written by Moritz Sperling
% 
% Licensed under the MIT License (see LICENSE for details)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

% input: original
input_real = '/data/experiment_root_real/eval/eval_adv';

% input: flow
input_flow = '/data/experiment_root_flow/eval/eval_adv';

% settings
global windowSize thresh;
thresh = 0.5;
windowSize = 5; 

% go
build_gui(input_real, input_flow);

% gui
function build_gui(input_real, input_flow)

    % get all data file names
    [files_real, files_flow, folders_real, folders_flow] = get_fnames(input_real, input_flow)
    
    % make sure data is ok
    if (sum(size(files_real) ==  size(files_flow)) == 2)
        
        % build gui
        ss = get(0,'ScreenSize');
        ui = uifigure;
        ui.Position = [10 ss(4)-100 360 600];
        uilabel(ui, ...
                'Position', [10 575 80 22],...
                'Text', 'Select Folder: ');
        dd = uidropdown(ui, ...
            'Position', [10 555 340 22], ...
            'Items', folders_real, ...
            'Value', folders_real{1}, ...
            'ValueChangedFcn', @(dd,event) plot_selection(dd, files_real, files_flow, folders_real, folders_flow));
        
        % plot first entry
        plot_selection(dd, files_real, files_flow, folders_real, folders_flow);
    else
        disp('input error');
    end
end

% get data from selected set and plot that
function plot_selection(dd, files_real, files_flow, folders_real, folders_flow)

    % get index of set
    fol = dd.Value;
    idx = find(contains(folders_real',fol));
    
    % get data from set
    fname_real = strcat(folders_real{idx}, '/', files_real{idx});
    fname_flow = strcat(folders_flow{idx}, '/', files_flow{idx});
    [data_real, data_flow] = get_data(fname_real, fname_flow);
    
    % plot
    plot_data(data_real, data_flow);
end

% plot data
function plot_data(data_real, data_flow)
    global windowSize thresh;
    
    % adjust flowdata +1
    data_flow.pred_collisions = [0; data_flow.pred_collisions];
    data_flow.pred_steerings  = [0; data_flow.pred_steerings];
%     data_flow.real_labels     = [0; data_flow.real_labels];

    % prep data for plots
    x0 = 0:numel(data_flow.pred_collisions) - 1;
    f = (1/windowSize)*ones(1,windowSize);
    c1 = filter(f, 1, data_real.pred_collisions);
    c2 = filter(f, 1, data_flow.pred_collisions);
    s1 = filter(f, 1, data_real.pred_steerings);
    s2 = filter(f, 1, data_flow.pred_steerings);

    % create confusion matrices
    t1 = double(c1 > thresh);
    t2 = double(c2 > thresh);
    cm1 = confusionmat(data_real.real_labels, t1);
    cm2 = confusionmat(data_flow.real_labels, t2);
    cm3 = confusionmat(t1, t2);
    
    % fix confusion matrix if broken
    if numel(cm1) < 4
       cm1 = [cm1 0; 0 0]; 
    end
    if numel(cm2) < 4
       cm2 = [cm2 0; 0 0]; 
    end
    if numel(cm3) < 4
       cm2 = [cm3 0; 0 0]; 
    end
    
    % plot results
    ss = get(0,'ScreenSize');
    ff = figure(1);
    ff.Position = [400 0 (ss(3) - 400) ss(4)];

    % collisions
    subplot(3,3,[1 2])
    bar(data_real.real_labels, 'g', 'EdgeColor', 'none');
    hold on
    bar(data_real.pred_collisions, 'r', 'EdgeColor', 'none');
    plot(x0, c1, '-k', 'LineWidth', 2);
    plot(x0, t1, '--k', 'LineWidth', 2);
    hold off
    title('DroNet: Predictions and Labels (Collisions)');
    ylim([0 1]);
    xlabel('Image #');
    ylabel('Predicted Value');
    l1 = legend('Ground Truth', 'Predictions', 'Smoothed Predictions', 'Binarized Predictions');
    l1.Position = [0.5 0.94 0.125 0.06];
    
    subplot(3,3,[4 5])
    bar(data_flow.real_labels, 'g', 'EdgeColor', 'none');
    hold on
    bar(data_flow.pred_collisions, 'b', 'EdgeColor', 'none');
    plot(x0, c2, '-k', 'LineWidth', 2);
    plot(x0, t2, '--k', 'LineWidth', 2);
    hold off
    title('FlowDroNet: Predictions and Labels (Collisions)');
    ylim([0 1]);
    xlabel('Image #');
    ylabel('Predicted Value');
    
    subplot(3,3,3)
    cc = confusionchart(cm1, {'clear','collision'});
    cc.Title = "Comparison with Ground Truth";
    cc.YLabel = "Ground Truth";
    cc.XLabel = "DroNet";

    subplot(3,3,6)
    cc = confusionchart(cm2, {'clear','collision'});
    cc.Title = "Comparison with Ground Truth";
    cc.YLabel = "Ground Truth";
    cc.XLabel = "FlowDroNet";
    
    subplot(3,3,9)
    cc = confusionchart(cm3, {'clear','collision'});
    cc.Title = "Comparison DroNet / FlowDroNet";
    cc.YLabel = "DroNet";
    cc.XLabel = "FlowDroNet";
    
    subplot(3,3,[7 8])
    plot(x0, s1, '-r', 'LineWidth', 2);
    hold on
    plot(x0, s2, '-b', 'LineWidth', 2);
    plot(x0, data_real.pred_steerings, ':r', 'LineWidth', 2);
    plot(x0, data_flow.pred_steerings, ':b', 'LineWidth', 2);
    hold off
    title('Comparison of Predicted Steering Angles');
    ylim([-1.5 1.5]);
    xlim([0 max(x0)]);
    xlabel('Image #');
    ylabel('Normalized Steering Angle');
    l2 = legend('DroNet smoothed', 'FlowDroNet smoothed', 'DroNet raw', 'FlowDroNet raw');
    l2.Position = [0.5 0.02 0.125 0.06];
end

% get eval data filenames
function [files_real, files_flow, folders_real, folders_flow] = get_fnames(input_real, input_flow)
    fname = 'predicted_and_real_labels_model';
    finfo_real = dir(char(strcat(input_real, '/*/', fname, '*')));
    finfo_flow = dir(char(strcat(input_flow, '/*/', fname, '*')));
    files_real = {finfo_real.name};
    folders_real = {finfo_real.folder};
    files_flow = {finfo_flow.name};
    folders_flow = {finfo_flow.folder};
end

% extract data
function [data_real, data_flow] = get_data(fname_real, fname_flow)
    data_real = jsondecode(fileread(fname_real));
    data_flow = jsondecode(fileread(fname_flow));
end