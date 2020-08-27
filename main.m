clearvars;  clc; close all;

%% Define model parameters
hrf_est_params.kernel_length = 35;  %Time in seconds
hrf_est_params.sampling_rate = 2.12; %Sampling interval in seconds
hrf_est_params.ignore = ceil(hrf_est_params.kernel_length/hrf_est_params.sampling_rate);
hrf_est_params.regression_method = 'LSR'; %options: 'LSR', 'PLSR'
hrf_est_params.generalized_least_sq_method = 'gfls_AR'; %options: 'OLS', 'gfls', 'gfls_AR'
hrf_est_params.AR_lag = 0; %Number of time points to be used for modeling temporal autocorrelation
hrf_est_params.cv_folds = 2; %Number of blocks of data for cross validation
hrf_est_params.plsregress_options = statset('UseParallel',true); %Options for PLSR in case of being used
hrf_est_params.plsregress_ncomp = 3; %Number of PLS components to be used in PLSR. Max ncomp = Number of input functions.
hrf_est_params.model_order_L = 2; %Maximum number of Laguerre functions.
hrf_est_params.alpha = 0.5:0.1:1; %Range of alpha parameters (control Laguerre rate of decay)
hrf_est_params.mu = 2.5:0.5:5; %Range of mu parameters (control Laguerre time-to-peak)
hrf_est_params.resolve_svd_sign_ambiguity_method = 'pos_max_hrf_peak'; %options: 'pos_max_u_coeff', 'pos_first_nonzero_v','pos_max_v_coeff', 'pos_max_hrf_peak', 'pos_hrf_area'.

%% Main estimation block
load('Simulation/data.mat');
inp = input;
out = BOLD;

%Call main estimation function
results = estimation_funct( inp, out, hrf_est_params);

%%PLOTS

%Plot input data;
figure; 
subplot(4,1,1); plot(input(:,1)); legend('delta band');
subplot(4,1,2); plot(input(:,2)); legend('theta band');
subplot(4,1,3); plot(input(:,3)); legend('alpha band');
subplot(4,1,4); plot(input(:,4)); legend('beta band');

%Plot true output and estimated output
figure;
plot(results.output); hold on; plot(results.output_pred);
legend({'BOLD measurement', 'Estimated BOLD'});

%Plot true and predicted hemodyanic response functions
figure; 
plot(results.k1_est_times,hrf); hold on; plot(results.k1_est_times,results.k1_est);
legend({'True HRF', 'Original HRF'});
        