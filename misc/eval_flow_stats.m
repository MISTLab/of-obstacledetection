%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% eval_flow_stats.m
% 
% Tool to compare the statistics from a flow dataset (to be used with the 
% results of the get_flow_stats.py or output from prep. script).
% 
% Written by Moritz Sperling
% 
% Licensed under the MIT License (see LICENSE for details)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

global ax folder dd_opts dd_in_1 dd_in_2 dd_opt_items ylimits;

% location of the .mat files to inspect
folder = "/data/flow_stats/";
ylimits = 10;

% get stat files in folder (let's hope there are no other .mat files)
fsearch_name = folder + '*.mat';
fdata = dir(char(fsearch_name));
fnames = {fdata.name};

% build gui
ss = get(0, 'ScreenSize');
ui = uifigure;
ui.Name = 'Flow Stats Comparison Tool';
ui.Position = [200 150 1200 600];
ax = [uiaxes(ui, 'Position', [10 70 580 520]);
      uiaxes(ui, 'Position', [610 70 580 520])];
dd_opt_items = {'Min Mean Max XY', 'Min Mean Max Magnitude', 'Histograms X', 'Histograms Y'};
dd_opts = uidropdown(ui, ...
    'Position', [500 20 200 22], ...
    'Items', dd_opt_items, ...
    'Value', dd_opt_items{1}, ...
    'ValueChangedFcn', @(dd_opts, event) change_plot(dd_opts));
dd_in_1 = uidropdown(ui, ...
    'Position', [50 20 200 22], ...
    'Items', fnames, ...
    'Value', fnames{1}, ...
    'ValueChangedFcn', @(dd_in_1, event) change_data(dd_in_1, ax(1)));
dd_in_2 = uidropdown(ui, ...
    'Position', [950 20 200 22], ...
    'Items', fnames, ...
    'Value', fnames{2}, ...
    'ValueChangedFcn', @(dd_in_2, event) change_data(dd_in_2, ax(2)));

% initial plot
change_plot(dd_opts);

% callback for changing plot type in both plots
function change_plot(dd_opts)
    global ax folder dd_in_1 dd_in_2 dd_opt_items;
    
    % change both plots
    plot_selection(dd_opts, ax(1), folder, dd_in_1.Value, dd_opt_items)
    plot_selection(dd_opts, ax(2), folder, dd_in_2.Value, dd_opt_items)
end

% callback for changing data in one of the plots
function change_data(dd_in, ax)
    global folder dd_opts dd_opt_items;
    
    % change plot
    plot_selection(dd_opts, ax, folder, dd_in.Value, dd_opt_items)
end

% select what to plot
function plot_selection(dd_opts, ax, folder, input, dd_opt_items)

    % load data
    stats = load(folder + input); 
        
    % call selected plot
    if strcmp(dd_opts.Value, dd_opt_items{1})
        plot_mmm(ax, stats, input);
    elseif strcmp(dd_opts.Value, dd_opt_items{2})
        plot_mag(ax, stats, input);
    elseif strcmp(dd_opts.Value, dd_opt_items{3})
        plot_hist_data(ax, stats, input, 'x');
    elseif strcmp(dd_opts.Value, dd_opt_items{4})
        plot_hist_data(ax, stats, input, 'y');
    end
end

% plot max min mean stats
function plot_mmm(ax, stats, titlestr)
    global ylimits;
    plot(ax, stats.max_x, ".r", 'LineWidth', 2);
    hold(ax, 'on');
    plot(ax, stats.max_y, ".g", 'LineWidth', 2);
    plot(ax, stats.min_x, ".m", 'LineWidth', 2);
    plot(ax, stats.min_y, ".c", 'LineWidth', 2);
    plot(ax, stats.mean_x, ".b", 'LineWidth', 2);
    plot(ax, stats.mean_y, ".y", 'LineWidth', 2);
    plot(ax, [0 numel(stats.max_x)], [mean(stats.max_x) mean(stats.max_x)], '--k', 'LineWidth', 2);
    plot(ax, [0 numel(stats.max_y)], [mean(stats.max_y) mean(stats.max_y)], ':k', 'LineWidth', 2);
    plot(ax, [0 numel(stats.min_x)], [mean(stats.min_x) mean(stats.min_x)], '--k', 'LineWidth', 2);
    plot(ax, [0 numel(stats.min_y)], [mean(stats.min_y) mean(stats.min_y)], ':k', 'LineWidth', 2);
    ylim(ax, [-ylimits ylimits]);
    xlim(ax, [0 numel(stats.max_x)]);
    legend(ax, "Max. x", "Max. y", "Min. x", "Min. y", "Mean x", "Mean y", "Mean Max. x", "Mean Max. y",'Location', 'northeastoutside');
    title(ax, "Min./Mean/Max. of X/Y (" + titlestr + ")", 'Interpreter', 'none');
    hold(ax, 'off');
end

% plot magnitude stats
function plot_mag(ax, stats, titlestr)
    global ylimits;
    plot(ax, stats.max_mag, ".r", 'LineWidth', 2);
    hold(ax, 'on');
    plot(ax, stats.mean_mag, ".b", 'LineWidth', 2);
    plot(ax, stats.min_mag, ".g", 'LineWidth', 2);
    plot(ax, [0 numel(stats.mean_mag)], [mean(stats.mean_mag) mean(stats.mean_mag)], ':k', 'LineWidth', 2);
    plot(ax, [0 numel(stats.max_mag)], [mean(stats.max_mag) mean(stats.max_mag)], '--k', 'LineWidth', 2);
    ylim(ax, [0 2 * ylimits]);
    xlim(ax, [0 numel(stats.max_x)]);
    legend(ax, "Max. Mag.", "Mean Mag.", "Min. Mag.", "Mean Mag.", "Mean Max. Mag", 'Location', 'northeastoutside');
    title(ax, "Min./Mean/Max. Magnitude (" + titlestr + ")", 'Interpreter', 'none');
    hold(ax, 'off');
end

% plot histogram samples
function plot_hist_data(ax, stats, titlestr, xy)
    h = double(stats.("histogram_" + xy)) / double(stats.n_pix);
    b = double(stats.("bins_" + xy));
    t = b(:, 1:end-1);
    
    [y, idx] = datasample(h(:), 10000);
    x = t(idx);
    
    plot(ax, x(:), y(:), '.k');
    hold(ax, 'on');
    ylim(ax, [0 0.1]);
    xlim(ax, [-50 50]);
    title(ax, "Histograms " + xy + " (" + titlestr + ")", 'Interpreter', 'none');
    legend(ax, 'off');
    hold(ax, 'off');
end
