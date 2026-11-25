%% =========================================================================
% Print_All_Results.m
% 集中统计并打印所有实验结果（含 5 个随机种子的均值与标准差 Mean ± Std）
% 覆盖 Table II (250s 多项式), Table III (750s 多项式), Table IV (96s 字符)
% 方法统一顺序:
% 1. FFN
% 2. DeLaN
% 3. CAD (名义物理参数)
% 4. SID (最小二乘辨识)
% 5. VDCDL-Fixed (消融 1: 惯性参数固定为 CAD，仅学习摩擦)
% 6. VDCDL-NoCADPrior (消融 2: 无 CAD 硬边界与经验排序约束)
% 7. VDCDL (完整所提方法)
% =========================================================================

clear; clc;
Plot_Path = fileparts(mfilename('fullpath')); if isempty(Plot_Path), Plot_Path = pwd; end
Path = fullfile(Plot_Path, '..', 'DVDPL', 'Data');

ffn_seeds = {'42', '518772', '942385', '607344', '149710'};
delan_seeds = {'43528', '246957', '783196', '760674', '31356'};
vdcdl_seeds = {'78963', '103809', '483628', '699028', '743384'};

%% -------------------------------------------------------------------------
% 1. TABLE II: 多项式测试 (185s) - 全方法基于 250s 训练集
% -------------------------------------------------------------------------
fprintf('\n====================================================================================\n');
fprintf('TABLE II: All-on-250s Benchmark (Polynomial Test 185s) [Mean ± Std]\n');
fprintf('====================================================================================\n');
fprintf('%-18s | %-26s | %-26s | %-26s\n', 'Method', 'Joint 1 (MAE/RMSE)', 'Joint 2 (MAE/RMSE)', 'Joint 3 (MAE/RMSE)');
fprintf('------------------------------------------------------------------------------------\n');

load(fullfile(Path, 'Data_Test_185s.mat'));
Test_real_185 = data.Ff;

% 1.1 FFN (250s)
preds = cell(5, 1);
for k = 1:5, preds{k} = readmatrix(fullfile(Plot_Path, sprintf('FFN_Test1_185s_%s.xlsx', ffn_seeds{k}))); end
[m_m, m_s, r_m, r_s] = calc_stats(preds, Test_real_185);
print_row('FFN', m_m, m_s, r_m, r_s);

% 1.2 DeLaN (250s)
for k = 1:5, preds{k} = readmatrix(fullfile(Plot_Path, sprintf('DeLaN_Test1_185s_%s.xlsx', delan_seeds{k}))); end
[m_m, m_s, r_m, r_s] = calc_stats(preds, Test_real_185);
print_row('DeLaN', m_m, m_s, r_m, r_s);

% 1.3 CAD (名义参数)
temp = load(fullfile(Plot_Path, 'CAD_Test_185s.mat'));
[m_m, r_m] = calc_single(temp.F_CAD, Test_real_185);
print_single_row('CAD', m_m, r_m);

% 1.4 SID (250s)
temp = load(fullfile(Plot_Path, 'LSQ_Test1_185s.mat'));
[m_m, r_m] = calc_single(temp.Fpre, Test_real_185);
print_single_row('SID', m_m, r_m);

% 1.5 VDCDL-Fixed (250s)
for k = 1:5, preds{k} = readmatrix(fullfile(Plot_Path, sprintf('VDCDL_Test1_185s_fixed_%s.xlsx', vdcdl_seeds{k}))); end
[m_m, m_s, r_m, r_s] = calc_stats(preds, Test_real_185);
print_row('VDCDL-Fixed', m_m, m_s, r_m, r_s);

% 1.6 VDCDL-NoCADPrior (250s)
for k = 1:5, preds{k} = readmatrix(fullfile(Plot_Path, sprintf('NoCADPrior_Test1_185s_%s.xlsx', vdcdl_seeds{k}))); end
[m_m, m_s, r_m, r_s] = calc_stats(preds, Test_real_185);
print_row('VDCDL-NoCADPrior', m_m, m_s, r_m, r_s);

% 1.7 VDCDL (250s)
for k = 1:5, preds{k} = readmatrix(fullfile(Plot_Path, sprintf('VDCDL_Test1_185s_%s.xlsx', vdcdl_seeds{k}))); end
[m_m, m_s, r_m, r_s] = calc_stats(preds, Test_real_185);
print_row('VDCDL', m_m, m_s, r_m, r_s);


%% -------------------------------------------------------------------------
% 2. TABLE III: 多项式测试 (185s) - 全方法基于 750s 训练集
% -------------------------------------------------------------------------
fprintf('\n====================================================================================\n');
fprintf('TABLE III: All-on-750s Benchmark (Polynomial Test 185s) [Mean ± Std]\n');
fprintf('====================================================================================\n');
fprintf('%-18s | %-26s | %-26s | %-26s\n', 'Method', 'Joint 1 (MAE/RMSE)', 'Joint 2 (MAE/RMSE)', 'Joint 3 (MAE/RMSE)');
fprintf('------------------------------------------------------------------------------------\n');

% 2.1 FFN (750s)
for k = 1:5, preds{k} = readmatrix(fullfile(Plot_Path, sprintf('FFN_Test2_185s_%s.xlsx', ffn_seeds{k}))); end
[m_m, m_s, r_m, r_s] = calc_stats(preds, Test_real_185);
print_row('FFN', m_m, m_s, r_m, r_s);

% 2.2 DeLaN (750s)
for k = 1:5, preds{k} = readmatrix(fullfile(Plot_Path, sprintf('DeLaN_Test2_185s_%s.xlsx', delan_seeds{k}))); end
[m_m, m_s, r_m, r_s] = calc_stats(preds, Test_real_185);
print_row('DeLaN', m_m, m_s, r_m, r_s);

% 2.3 CAD (名义参数)
temp = load(fullfile(Plot_Path, 'CAD_Test_185s.mat'));
[m_m, r_m] = calc_single(temp.F_CAD, Test_real_185);
print_single_row('CAD', m_m, r_m);

% 2.4 SID (750s)
temp = load(fullfile(Plot_Path, 'LSQ_Test2_185s.mat'));
[m_m, r_m] = calc_single(temp.Fpre, Test_real_185);
print_single_row('SID', m_m, r_m);

% 2.5 VDCDL-Fixed (750s)
for k = 1:5, preds{k} = readmatrix(fullfile(Plot_Path, sprintf('VDCDL_Test2_185s_fixed_%s.xlsx', vdcdl_seeds{k}))); end
[m_m, m_s, r_m, r_s] = calc_stats(preds, Test_real_185);
print_row('VDCDL-Fixed', m_m, m_s, r_m, r_s);

% 2.6 VDCDL-NoCADPrior (750s)
for k = 1:5, preds{k} = readmatrix(fullfile(Plot_Path, sprintf('NoCADPrior_Test2_185s_%s.xlsx', vdcdl_seeds{k}))); end
[m_m, m_s, r_m, r_s] = calc_stats(preds, Test_real_185);
print_row('VDCDL-NoCADPrior', m_m, m_s, r_m, r_s);

% 2.7 VDCDL (750s)
for k = 1:5, preds{k} = readmatrix(fullfile(Plot_Path, sprintf('VDCDL_Test2_185s_%s.xlsx', vdcdl_seeds{k}))); end
[m_m, m_s, r_m, r_s] = calc_stats(preds, Test_real_185);
print_row('VDCDL', m_m, m_s, r_m, r_s);


%% -------------------------------------------------------------------------
% 3. TABLE IV: 字符绘制测试 (96s)
% -------------------------------------------------------------------------
fprintf('\n====================================================================================\n');
fprintf('TABLE IV: Character Drawing Test (96s) [Mean ± Std]\n');
fprintf('====================================================================================\n');
fprintf('%-18s | %-26s | %-26s | %-26s\n', 'Method', 'Joint 1 (MAE/RMSE)', 'Joint 2 (MAE/RMSE)', 'Joint 3 (MAE/RMSE)');
fprintf('------------------------------------------------------------------------------------\n');

load(fullfile(Path, 'Data_Test_draw8_96s.mat'));
Test_real_96 = data.Ff;

% 3.1 FFN (750s)
for k = 1:5, preds{k} = readmatrix(fullfile(Plot_Path, sprintf('FFN_Test3_96s_%s.xlsx', ffn_seeds{k}))); end
[m_m, m_s, r_m, r_s] = calc_stats(preds, Test_real_96);
print_row('FFN(750s)', m_m, m_s, r_m, r_s);

% 3.2 DeLaN (750s)
for k = 1:5, preds{k} = readmatrix(fullfile(Plot_Path, sprintf('DeLaN_Test3_96s_%s.xlsx', delan_seeds{k}))); end
[m_m, m_s, r_m, r_s] = calc_stats(preds, Test_real_96);
print_row('DeLaN(750s)', m_m, m_s, r_m, r_s);

% 3.3 CAD (名义参数)
temp = load(fullfile(Plot_Path, 'CAD_Test_96s.mat'));
[m_m, r_m] = calc_single(temp.F_CAD, Test_real_96);
print_single_row('CAD', m_m, r_m);

% 3.4 SID (750s)
temp = load(fullfile(Plot_Path, 'LSQ_Test2_96s.mat'));
[m_m, r_m] = calc_single(temp.Fpre, Test_real_96);
print_single_row('SID(750s)', m_m, r_m);

% 3.5 VDCDL-Fixed (250s)
for k = 1:5, preds{k} = readmatrix(fullfile(Plot_Path, sprintf('VDCDL_Test3_96s_fixed_%s.xlsx', vdcdl_seeds{k}))); end
[m_m, m_s, r_m, r_s] = calc_stats(preds, Test_real_96);
print_row('VDCDL-Fixed(250s)', m_m, m_s, r_m, r_s);

% 3.6 VDCDL-NoCADPrior (250s)
for k = 1:5, preds{k} = readmatrix(fullfile(Plot_Path, sprintf('NoCADPrior_Test3_96s_%s.xlsx', vdcdl_seeds{k}))); end
[m_m, m_s, r_m, r_s] = calc_stats(preds, Test_real_96);
print_row('VDCDL-NoCADPrior(250s)', m_m, m_s, r_m, r_s);

% 3.7 VDCDL (250s)
for k = 1:5, preds{k} = readmatrix(fullfile(Plot_Path, sprintf('VDCDL_Test3_96s_%s.xlsx', vdcdl_seeds{k}))); end
[m_m, m_s, r_m, r_s] = calc_stats(preds, Test_real_96);
print_row('VDCDL(250s)', m_m, m_s, r_m, r_s);

fprintf('====================================================================================\n\n');

%% 辅助函数
function [mae_mean, mae_std, rmse_mean, rmse_std] = calc_stats(preds, ground_truth)
    n_seeds = length(preds);
    maes = zeros(n_seeds, 3);
    rmses = zeros(n_seeds, 3);
    for k = 1:n_seeds
        err = preds{k} - ground_truth;
        maes(k, :) = mean(abs(err), 1);
        rmses(k, :) = sqrt(mean(err.^2, 1));
    end
    mae_mean = mean(maes, 1);
    mae_std = std(maes, 0, 1);
    rmse_mean = mean(rmses, 1);
    rmse_std = std(rmses, 0, 1);
end

function [mae, rmse] = calc_single(pred, ground_truth)
    err = pred - ground_truth;
    mae = mean(abs(err), 1);
    rmse = sqrt(mean(err.^2, 1));
end

function print_row(name, mae_m, mae_s, rmse_m, rmse_s)
    str_j1 = sprintf('%5.2f±%-5.2f / %5.2f±%-5.2f', mae_m(1), mae_s(1), rmse_m(1), rmse_s(1));
    str_j2 = sprintf('%5.2f±%-5.2f / %5.2f±%-5.2f', mae_m(2), mae_s(2), rmse_m(2), rmse_s(2));
    str_j3 = sprintf('%5.2f±%-5.2f / %5.2f±%-5.2f', mae_m(3), mae_s(3), rmse_m(3), rmse_s(3));
    fprintf('%-22s | %-26s | %-26s | %-26s\n', name, str_j1, str_j2, str_j3);
end

function print_single_row(name, mae_m, rmse_m)
    str_j1 = sprintf('%5.2f        / %5.2f       ', mae_m(1), rmse_m(1));
    str_j2 = sprintf('%5.2f        / %5.2f       ', mae_m(2), rmse_m(2));
    str_j3 = sprintf('%5.2f        / %5.2f       ', mae_m(3), rmse_m(3));
    fprintf('%-22s | %-26s | %-26s | %-26s\n', name, str_j1, str_j2, str_j3);
end
