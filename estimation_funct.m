function [output_args] = estimation_funct( input, output, hrf_est_params)
%UNTITLED7 Summary of this function goes here
%Detailed explanation goes here

%Parameters
ALPHA = hrf_est_params.alpha;
MU    = hrf_est_params.mu;
num_of_basis_functs = hrf_est_params.model_order_L;
kernel_length = hrf_est_params.kernel_length;
%sampling_rate = hrf_est_params.VTrigs;
sampling_rate = hrf_est_params.sampling_rate;
ignore = hrf_est_params.ignore;
AR_lag = hrf_est_params.AR_lag;
regression_method = hrf_est_params.regression_method;
generalized_least_sq_method = hrf_est_params.generalized_least_sq_method;
cv_folds = hrf_est_params.cv_folds;
plsregress_options = hrf_est_params.plsregress_options;
plsregress_ncomp = hrf_est_params.plsregress_ncomp;
sign_ambiguity_method = hrf_est_params.resolve_svd_sign_ambiguity_method;
%---

%Compute Nomralized Mean Squarres Errors using Cross Validation
%and define optimal model as the one that minimizes mean mse.
optimal_mean_mse = inf; %Initialize optimal mse to inf;
for a = 1:length(ALPHA)
    for m = 1:length(MU)
        
        alpha = ALPHA(a);
        mu = MU(m);
        sigma = 1.5;
        [Vmat, basis_set, basis_set_sampling_rate] = Vmat_construction( input, alpha,  mu, sigma, num_of_basis_functs, sampling_rate, kernel_length );
        %        n    =length(output(ignore:end));
%        k_fold_cv_res = k_fold_cv( Vmat(ignore:end,:), output(ignore:end,:), cv_folds, plsregress_options, plsregress_ncomp, generalized_least_sq_method, regression_method, AR_lag );
        k_fold_cv_res = k_fold_cv_conv_err(input(ignore:end,:), Vmat(ignore:end,:), output(ignore:end,:), sampling_rate, cv_folds, plsregress_options, plsregress_ncomp, generalized_least_sq_method, regression_method, AR_lag, basis_set, basis_set_sampling_rate);
        k_fold_cv_res.mu = mu;
        k_fold_cv_res.alpha = alpha;
        k_fold_cv_res.sigma = sigma;
        
        %Select optimal model params based on minimum k_fold_cv mse
        if optimal_mean_mse > k_fold_cv_res.mean_mse
            optimal_mean_mse  = k_fold_cv_res.mean_mse;
            optimal_model_params = k_fold_cv_res;
        end
    end
end


%Estimate Regressors using the optimal Laguerre parameters
alpha = optimal_model_params.alpha;
mu = optimal_model_params.mu;
sigma = optimal_model_params.sigma;
[Vmat, basis_set, basis_set_sampling_rate] = Vmat_construction( input, alpha,  mu, sigma, num_of_basis_functs, sampling_rate, kernel_length );
%n    =length(output(ignore:end));

%Detrend the data before used in model parameter estimation
% inp = zscore(Vmat);
% out = zscore(output);
Vmat_inp = Vmat;
out = output;
make_plot =0;
[Vmat_inp, out] = data_detrend( Vmat_inp, out, make_plot ); %Demean i/o data

%Get HRF estimate, output model prediction, and statistics for the optimal
%model
prediction_results =  main_estimation_funct(Vmat_inp,input,out,sampling_rate,plsregress_options, plsregress_ncomp, cv_folds, generalized_least_sq_method, regression_method, AR_lag, basis_set, basis_set_sampling_rate, sign_ambiguity_method);

%Find the Laguerre and input (EEG) regression parameters using SVD.
regress_params = prediction_results.C_coeff_full_model(2:end);
if size(input,2)>1
    Cpar                         = reshape(regress_params,size(input,2),size(basis_set,2));
    [U_old,S,V_old]              = svd(Cpar,'econ');
    
    %Check if model prediction is not significantly different that null
    %prediction: i.e. chech is C_coeff == 0  or equivalently if S == 0.
    if isempty(find(S, 1))
        U_old = zeros(size(U_old));
        V_old = zeros(size(V_old));
    end
    %Resolve SVD sign ambiguity.
    [U_new, VS_new] = resolve_sign_ambiguity(U_old, S, V_old, Cpar, basis_set, sign_ambiguity_method);
    laguerre_regress_coeff = VS_new(:,1);
    inp_regress_coeff = U_new(:,1);
elseif size(input,2)==1
    laguerre_regress_coeff = regress_params;
    inp_regress_coeff = 1;
else
    error('Error: the number of input vectors is something weird!!! Check it out!!!');
end

k1_est  = basis_set*laguerre_regress_coeff;
inp_est = input*inp_regress_coeff;
inp_est = inp_est(ignore:end);
output  = output(ignore:end);

%output_args
output_args                                 = optimal_model_params;
output_args.num_of_basis_functs             = num_of_basis_functs;
output_args.ar_model_ord_for_noise          = AR_lag;
output_args.inp_coeff_a                     = inp_regress_coeff;
output_args.lag_coeff_b                     = laguerre_regress_coeff;
if strcmp(regression_method,'PLSR')
output_args.PLSR_inner_coeff                = prediction_results.PLSR_inner_coeff;
end
output_args.inp_est                         = inp_est;
output_args.output_pred                     = prediction_results.output_pred(ignore:end);
output_args.output_pred_tot                 = prediction_results.output_pred_tot(ignore:end);
output_args.output                          = output;
output_args.k0_est                          = prediction_results.output_pred_dc;
output_args.k1_est                          = k1_est;
output_args.k1_est_srate                    = basis_set_sampling_rate;
output_args.k1_est_times                    = 0:output_args.k1_est_srate:output_args.k1_est_srate*length(output_args.k1_est)-output_args.k1_est_srate;
% output_args.bic                             = stat_diagnostics.bic;
% output_args.aic                             = stat_diagnostics.aic;
% output_args.r2                              = stat_diagnostics.r2;
% output_args.r2adj                           = stat_diagnostics.r2adj;
return
function [ dt_input, dt_output] = data_detrend( Input, Output, make_plot )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

if make_plot
    figure;
    subplot(2,1,1)
    plot(0:length(Output)-1,Output), grid
    hold on;
    plot(0:length(Input(:,1))-1,Input(:,1))
    title 'Original Signals'
    if size(Input,2)>1
        legend('First input signal');
    end
end

data = Output;
data = data-mean(data);
% data = detrend(data);
% dt_output = data;
% data = detrend(data);
opol = 3;
t = (1:length(data))';
[p,s,mu] = polyfit(t,data,opol);
f_y = polyval(p,t,[],mu);
dt_output = data - f_y;


for i=1:size(Input,2)
    data = Input(:,i);
    data = data-mean(data);
%     data = detrend(data);
%     dt_input(:,i) = data-mean(data);
    data = detrend(data);
    opol = 3;
    t = (1:length(data))';
    [p,s,mu] = polyfit(t,data,opol);
    f_y = polyval(p,t,[],mu);
    dt_input(:,i) = data - f_y;

end

if make_plot
    subplot(2,1,2)
    plot(0:length(dt_output)-1,dt_output), grid
    hold on;
    plot(0:length(dt_input(:,1))-1,dt_input(:,1))
    title 'Detrended Signals'
    if size(Input,2)>1
        legend('First input signal');
    end
end

return
function [Vmat, basis, basis_sampling_rate] = Vmat_construction( input, alpha, mu, sigma, num_of_basis_functs, fMRI_sampling_rate, kernel_length )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
%---------
bf_sampling_rate = 0.1;
[basis,basis_sampling_rate] = laguerre_basis_smothed_with_gaussian_kernel( alpha, mu, sigma, num_of_basis_functs, bf_sampling_rate, kernel_length );
[N,D] = rat(fMRI_sampling_rate/basis_sampling_rate);

%x  = times of data points
%v  = values of data points at x
%xq = times of query datapoints
x  = 0:2.12:2.12*(length(input)-1);
xq = 0:0.1:2.12*(length(input)-1);

for i=1:size(input,2)
    
input_q(:,i) = interp1(x,input(:,i),xq,'linear');

end

count=1;
for i=1:size(basis,2)
    for j=1:size(input,2)
        V(:,count) = conv(input_q(:,j),basis(:,i));
        count = count+1;
    end
end
Vmat_upsampled = V(1:end-length(basis)+1,:);
for k=1:size(Vmat_upsampled,2)
    Vmat(:,k) = resample(Vmat_upsampled(:,k),D,N);
end
return
function [ output_args, sampling_rate] = laguerre_basis_smothed_with_gaussian_kernel( alpha, mu, sigma, num_of_basis_functs, sampling_rate, kernel_length )
%laguerre_basis_smothed_with_gaussian_kernel construct Sperical Laguerre Basis Functions.
%The sampling rate of these functions is 10Hz to avoid significant distortion of the constructed
%basis functions due to aliasing or antializing filtering. Therefore, input
%singals convolved with these basis functions must be at 10Hz. The Vmat
%matrix must be doqnsampled at the fMRI TR. (Use the resample function that employes
%antialising). Filtering the Vmat matrix potentially distorts singals in
%lesser extend compared to the distortion of the basis funcitions (need to
%check this)...
%   Detailed explanation goes here

tau  = alpha;
P    = num_of_basis_functs; % # of Laguerre functions
time = 0:sampling_rate:kernel_length-1;
s = sigma;

%Generate spherical Laguerre basis
for p=0:P-1
    for i=1:length(time)
        r = time(i);
        K(i,p+1)  = sqrt(factorial(p)./factorial(p+2)).*(exp(-r./(2.*tau))./(sqrt(tau^(3)))).*generalized_laguerre(2, p, r./tau);
        rK(i,p+1) = r.*K(i,p+1) ;
    end
end

%Smooth using Gausian kernel N(mu,sigma=1)
for p=0:P-1
    gK_dummy  = conv(rK(:,p+1),gaussmf(time,[s,mu])); %[1,mu]
    gK(:,p+1) = gK_dummy(1:end-length(gaussmf(time,[s,mu]))+1);
end

%Apply Gram-Schmid orthogonalization
%Multiply with 1/sampling rate to obtain the correct scaling of the Laguerre functions
if size(gK,2)==1
    output_args = gK*(sampling_rate)^(-1);
else
    output_args = GramSchmidt(gK)*(sampling_rate)^(-1);
end

return
function [L_out] = generalized_laguerre(a,p,r)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

out = 0;
for j=0:p
    out = out + nchoosek(p+a,p-j).*((-r)^j)./factorial(j);
end
L_out = out;

return
function vect = GramSchmidt(vect)
k = size(vect,2);
if k<2
    error("Error: The input of the GramSchmidt function must include at least two vectors.")
end

for i = 1:1:k
    vect(:,i) = vect(:,i) / norm(vect(:,i));
    for j = i+1:1:k
        %Estimate projection
        projection = (dot(vect(:,j),vect(:,i)) / dot(vect(:,i),vect(:,i))) * vect(:,i);
        vect(:,j) = vect(:,j) - projection;
    end
    vect(:,i) = vect(:,i) / norm(vect(:,i));
end
return
function [ PLSR_params, stat_diagnostics, flag ] = pls_prokopis( inp,out,plsregress_ncomp,generalized_least_sq_method,AR_lag,plsregress_cv_folds,plsregress_options,return_stats)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%Get Y loading and X scores using MATLAB's plsregress algorithm
[~,YL,XS,~,~,~,~,stats] = plsregress(inp,out,plsregress_ncomp);%,'cv',plsregress_cv_folds,'options',plsregress_options);
for i=1:length(YL)
    pls_inp(:,i) = XS(:,i)*YL(i);
end
%pls_inp_intercept = [ones(size(pls_inp,1),1), pls_inp];
pls_out = out;
%Estimate for PLSR coreffients BETA in equation (25) of De Jong et al.,
%1993 paper "SIMPLS: an alternative approach to PLSR"
switch generalized_least_sq_method
    case 'OLS'
        mdl = fitlm(pls_inp,pls_out);                    %PLSR coefficients for the inner relation.
        %mdl = fitlm(pls_inp,pls_out,'Intercept',false);
        Cmat = mdl.Coefficients.Estimate(2:end);
        Cmat = [mean(pls_out)-mean(pls_inp)*Cmat(1:end); Cmat(1:end)];
        output_pred = [ones(length(pls_inp),1) pls_inp]*Cmat;
        
        if return_stats ==1
            %stat_diagnostics = mdl;
            
            n = length(pls_out);
            p = length(Cmat);
            
            stat_diagnostics.CoefficientCovariance  = mdl.CoefficientCovariance;
            stat_diagnostics.ParamStandardError  = mdl.Coefficients.SE;
            stat_diagnostics.ParamVals  = Cmat;
            stat_diagnostics.NumEstimatedCoefficients = p;
            stat_diagnostics.NumObservations = n;
            stat_diagnostics.DFE  = stat_diagnostics.NumObservations-stat_diagnostics.NumEstimatedCoefficients; %Degrees of freedom for errors
            stat_diagnostics.DFR  = stat_diagnostics.NumEstimatedCoefficients-1; %Degrees of freedom for regression model prediction
            stat_diagnostics.DFT  = stat_diagnostics.NumObservations-1; %Degrees of freedom for total model
            stat_diagnostics.SSE  = (norm(pls_out-output_pred))^2; %Sum of Squerres for error
            stat_diagnostics.MSE  = stat_diagnostics.SSE/(stat_diagnostics.DFE); %Mean sum of Squerres for error
            stat_diagnostics.SSR  = (norm(output_pred-mean(output_pred)))^2; %This is how matlab defines Regression Sum of Squerres (e.g. help fitglm)
            stat_diagnostics.MSR  = stat_diagnostics.SSR/stat_diagnostics.DFR; %Mean Regression Sum of Squerres
            stat_diagnostics.SST  = (norm(pls_out-mean(pls_out)))^2; %This is how matlab defines Total Sum of Squerres (e.g. help fitglm)
            stat_diagnostics.MST  = stat_diagnostics.SST/stat_diagnostics.DFT; %Mean Regression Sum of Squerres
            stat_diagnostics.NMSE = stat_diagnostics.SSE/stat_diagnostics.SST;
            for i=1:p
                stat_diagnostics.Params_tStat(i)  = stat_diagnostics.ParamVals(i)/stat_diagnostics.ParamStandardError(i);
                stat_diagnostics.Params_tPval(i) = (1-tcdf(abs(stat_diagnostics.Params_tStat(i)),stat_diagnostics.DFE))*2;
            end
            stat_diagnostics.Model_fStat  = (stat_diagnostics.SSR/stat_diagnostics.DFR)/(stat_diagnostics.SSE/stat_diagnostics.DFE);
            stat_diagnostics.Model_fPval = 1 - fcdf(stat_diagnostics.Model_fStat,stat_diagnostics.DFR,stat_diagnostics.DFE);
            [stat_diagnostics.r2,stat_diagnostics.r2adj] =rsquared(pls_out,output_pred,p);
            stat_diagnostics.loglik    = -.5*stat_diagnostics.NumObservations*(log(2*pi) + log(stat_diagnostics.SSE/stat_diagnostics.NumObservations) + 1);
            [stat_diagnostics.aic,stat_diagnostics.bic] = aicbic(stat_diagnostics.loglik,stat_diagnostics.NumEstimatedCoefficients,stat_diagnostics.NumObservations);
            %Check parameter estimation diagnostics
            if stat_diagnostics.Model_fPval<0.05
                
                %Set to 0 all params if there associated pvalue is > 0.05
                for p=1:length(Cmat)
                    if stat_diagnostics.Params_tPval(p)>0.05
                        Cmat(p)=0;
                    end
                end
                %Estimate the PLSR coeffs for the outer relation using only the
                %significant coeffs in Cmat.
                C2   =  stats.W*diag(Cmat(2:end))*YL';                      %PLSR coefficients for the inner relation.
                %Notice that here only the 2:end values of the Cmat are
                %considered, as the intercept term (i.e. mean(pls_out)) is
                %not included in eqn (24) of De Jong S. et. al., 1933 (SIMPLS paper).
                C2_with_intercept = [ mean(out) - mean(inp,1)*C2; C2];      %PLSR coefficients for the outer relation.
                PLSR_params = C2_with_intercept;
                flag = 0; %Used to indicated that the coeff estimates are reliable
            else
                %Set to 0 all params
                for p=1:length(Cmat)
                    if stat_diagnostics.Params_tPval(p)>0.05
                        Cmat(p)=0;
                    end
                end
                %Estimate the PLSR coeffs for the outer relation using only the
                %significant coeffs in Cmat.
                C2   =  stats.W*diag(Cmat(2:end))*YL';                      %PLSR coefficients for the inner relation.
                %Notice that here only the 2:end values of the Cmat are
                %considered, as the intercept term (i.e. mean(pls_out)) is
                %not included in eqn (24) of De Jong S. et. al., 1933 (SIMPLS paper).
                C2_with_intercept = [ mean(out) - mean(inp,1)*C2; C2];      %PLSR coefficients for the outer relation.
                PLSR_params = zeros(size(C2_with_intercept));
                flag = 1; %Used to indicated that the coeff estimates are NOT reliable
            end
        else
            stat_diagnostics = [];
        end
        
    case 'gfls'
        
        [Cmat,se,EstCoeffCov] = fgls(pls_inp,pls_out,'intercept',false,'innovMdl','AR','arLags',AR_lag,'numIter',10);
        Cmat = [mean(pls_out)-mean(pls_inp)*Cmat(1:end); Cmat(1:end)];
        output_pred = [ones(length(pls_inp),1) pls_inp]*Cmat;
        
        if return_stats ==1
            n = length(pls_out);
            p = length(Cmat);
            
            stat_diagnostics.CoefficientCovariance  = EstCoeffCov;
            stat_diagnostics.ParamStandardError  = se;
            stat_diagnostics.ParamVals  = Cmat;
            stat_diagnostics.NumEstimatedCoefficients = p;
            stat_diagnostics.NumObservations = n;
            stat_diagnostics.DFE  = stat_diagnostics.NumObservations-stat_diagnostics.NumEstimatedCoefficients; %Degrees of freedom for errors
            stat_diagnostics.DFR  = stat_diagnostics.NumEstimatedCoefficients-1; %Degrees of freedom for regression model prediction
            stat_diagnostics.DFT  = stat_diagnostics.NumObservations-1; %Degrees of freedom for total model
            stat_diagnostics.SSE  = (norm(pls_out-output_pred))^2; %Sum of Squerres for error
            stat_diagnostics.MSE  = stat_diagnostics.SSE/(stat_diagnostics.DFE); %Mean sum of Squerres for error
            stat_diagnostics.SSR  = (norm(output_pred-mean(output_pred)))^2; %This is how matlab defines Regression Sum of Squerres (e.g. help fitglm)
            stat_diagnostics.MSR  = stat_diagnostics.SSR/stat_diagnostics.DFR; %Mean Regression Sum of Squerres
            stat_diagnostics.SST  = (norm(pls_out-mean(pls_out)))^2; %This is how matlab defines Total Sum of Squerres (e.g. help fitglm)
            stat_diagnostics.MST  = stat_diagnostics.SST/stat_diagnostics.DFT; %Mean Regression Sum of Squerres
            stat_diagnostics.NMSE = stat_diagnostics.SSE/stat_diagnostics.SST;
            for i=1:p
                stat_diagnostics.Params_tStat(i)  = stat_diagnostics.ParamVals(i)/stat_diagnostics.ParamStandardError(i);
                stat_diagnostics.Params_tPval(i) = (1-tcdf(abs(stat_diagnostics.Params_tStat(i)),stat_diagnostics.DFE))*2;
            end
            stat_diagnostics.Model_fStat  = (stat_diagnostics.SSR/stat_diagnostics.DFR)/(stat_diagnostics.SSE/stat_diagnostics.DFE);
            stat_diagnostics.Model_fPval = 1 - fcdf(stat_diagnostics.Model_fStat,stat_diagnostics.DFR,stat_diagnostics.DFE);
            [stat_diagnostics.r2,stat_diagnostics.r2adj] =rsquared(pls_out,output_pred,p);
            stat_diagnostics.loglik    = -.5*stat_diagnostics.NumObservations*(log(2*pi) + log(stat_diagnostics.SSE/stat_diagnostics.NumObservations) + 1);
            [stat_diagnostics.aic,stat_diagnostics.bic] = aicbic(stat_diagnostics.loglik,stat_diagnostics.NumEstimatedCoefficients,stat_diagnostics.NumObservations);
            
            %Check parameter estimation diagnostics
            if stat_diagnostics.Model_fPval<0.05
                
                %Set to 0 all params if there associated pvalue is > 0.05
                for p=1:length(Cmat)
                    if stat_diagnostics.Params_tPval(p)>0.05
                        Cmat(p)=0;
                    end
                end
                %Estimate the PLSR coeffs for the outer relation using only the
                %significant coeffs in Cmat.
                C2   =  stats.W*diag(Cmat(2:end))*YL';                      %PLSR coefficients for the inner relation.
                %Notice that here only the 2:end values of the Cmat are
                %considered, as the intercept term (i.e. mean(pls_out)) is
                %not included in eqn (24) of De Jong S. et. al., 1933 (SIMPLS paper).
                C2_with_intercept = [ mean(out) - mean(inp,1)*C2; C2];      %PLSR coefficients for the outer relation.
                PLSR_params = C2_with_intercept;
                flag = 0; %Used to indicated that the coeff estimates are reliable
            else
                %Set to 0 all params
                for p=1:length(Cmat)
                    if stat_diagnostics.Params_tPval(p)>0.05
                        Cmat(p)=0;
                    end
                end
                %Estimate the PLSR coeffs for the outer relation using only the
                %significant coeffs in Cmat.
                C2   =  stats.W*diag(Cmat(2:end))*YL';                      %PLSR coefficients for the inner relation.
                %Notice that here only the 2:end values of the Cmat are
                %considered, as the intercept term (i.e. mean(pls_out)) is
                %not included in eqn (24) of De Jong S. et. al., 1933 (SIMPLS paper).
                C2_with_intercept = [ mean(out) - mean(inp,1)*C2; C2];      %PLSR coefficients for the outer relation.
                PLSR_params = zeros(size(C2_with_intercept));
                flag = 1; %Used to indicated that the coeff estimates are NOT reliable
            end
        end
        
        
    case 'gfls_AR'
        max_iter = 20;
        
        pls_inp_intercept = [ones(length(pls_inp),1) pls_inp];
        [Cmat, ~, se, EstCoeffCov] = gfls_AR_param_est(pls_inp_intercept,pls_out,AR_lag,max_iter);
        Cmat = [mean(pls_out)-mean(pls_inp,1)*Cmat(2:end); Cmat(2:end)];
        output_pred = pls_inp_intercept *Cmat;
        
        if return_stats ==1
            n = length(pls_out);
            p = length(Cmat);
            
            stat_diagnostics.CoefficientCovariance  = EstCoeffCov;
            stat_diagnostics.ParamStandardError  = se;
            stat_diagnostics.ParamVals  = Cmat;
            stat_diagnostics.NumEstimatedCoefficients = p;
            stat_diagnostics.NumObservations = n;
            stat_diagnostics.DFE  = stat_diagnostics.NumObservations-stat_diagnostics.NumEstimatedCoefficients; %Degrees of freedom for errors
            stat_diagnostics.DFR  = stat_diagnostics.NumEstimatedCoefficients-1; %Degrees of freedom for regression model prediction
            stat_diagnostics.DFT  = stat_diagnostics.NumObservations-1; %Degrees of freedom for total model
            stat_diagnostics.SSE  = (norm(pls_out-output_pred))^2; %Sum of Squerres for error
            stat_diagnostics.MSE  = stat_diagnostics.SSE/(stat_diagnostics.DFE); %Mean sum of Squerres for error
            stat_diagnostics.SSR  = (norm(output_pred-mean(output_pred)))^2; %This is how matlab defines Regression Sum of Squerres (e.g. help fitglm)
            stat_diagnostics.MSR  = stat_diagnostics.SSR/stat_diagnostics.DFR; %Mean Regression Sum of Squerres
            stat_diagnostics.SST  = (norm(pls_out-mean(pls_out)))^2; %This is how matlab defines Total Sum of Squerres (e.g. help fitglm)
            stat_diagnostics.MST  = stat_diagnostics.SST/stat_diagnostics.DFT; %Mean Regression Sum of Squerres
            stat_diagnostics.NMSE = stat_diagnostics.SSE/stat_diagnostics.SST;
            for i=1:p
                stat_diagnostics.Params_tStat(i)  = stat_diagnostics.ParamVals(i)/stat_diagnostics.ParamStandardError(i);
                stat_diagnostics.Params_tPval(i) = (1-tcdf(abs(stat_diagnostics.Params_tStat(i)),stat_diagnostics.DFE))*2;
            end
            stat_diagnostics.Model_fStat  = (stat_diagnostics.SSR/stat_diagnostics.DFR)/(stat_diagnostics.SSE/stat_diagnostics.DFE);
            stat_diagnostics.Model_fPval = 1 - fcdf(stat_diagnostics.Model_fStat,stat_diagnostics.DFR,stat_diagnostics.DFE);
            [stat_diagnostics.r2,stat_diagnostics.r2adj] =rsquared(pls_out,output_pred,p);
            stat_diagnostics.loglik    = -.5*stat_diagnostics.NumObservations*(log(2*pi) + log(stat_diagnostics.SSE/stat_diagnostics.NumObservations) + 1);
            [stat_diagnostics.aic,stat_diagnostics.bic] = aicbic(stat_diagnostics.loglik,stat_diagnostics.NumEstimatedCoefficients,stat_diagnostics.NumObservations);
            
            if stat_diagnostics.Model_fPval<0.05
                %Set to 0 all params if there associated pvalue is > 0.05
                for p=1:length(Cmat)
                    if stat_diagnostics.Params_tPval(p)>0.05
                        Cmat(p)=0;
                    end
                end
                %Estimate the PLSR coeffs for the outer relation using only the
                %significant coeffs in Cmat.
                C2   =  stats.W*diag(Cmat(2:end))*YL';                      %PLSR coefficients for the inner relation.
                %Notice that here only the 2:end values of the Cmat are
                %considered, as the intercept term (i.e. mean(pls_out)) is
                %not included in eqn (24) of De Jong S. et. al., 1933 (SIMPLS paper).
                C2_with_intercept = [ mean(out) - mean(inp,1)*C2; C2];      %PLSR coefficients for the outer relation.
                PLSR_params = C2_with_intercept;
                
                %                 %Importance selection for input functions
                %                 alpha_mc = 0.01;
                %                 acfopt = true;
                %                 Xm = bsxfun(@minus,inp,mean(inp)); % Mean centered X
                %                 [stat_diagnostics.smcF, stat_diagnostics.smcFcrit] = smc(PLSR_params(2:end), Xm,alpha_mc,acfopt); %Ignore intersept coeff
                flag = 0; %Used to indicated that the coeff estimates are reliable
            else
                %Set to 0 all params
                for p=1:length(Cmat)
                    if stat_diagnostics.Params_tPval(p)>0.05
                        Cmat(p)=0;
                    end
                end
                %Estimate the PLSR coeffs for the outer relation using only the
                %significant coeffs in Cmat.
                C2   =  stats.W*diag(Cmat(2:end))*YL';                      %PLSR coefficients for the inner relation.
                %Notice that here only the 2:end values of the Cmat are
                %considered, as the intercept term (i.e. mean(pls_out)) is
                %not included in eqn (24) of De Jong S. et. al., 1933 (SIMPLS paper).
                C2_with_intercept = [ mean(out) - mean(inp,1)*C2; C2];      %PLSR coefficients for the outer relation.
                PLSR_params = zeros(size(C2_with_intercept));
                flag = 1; %Used to indicated that the coeff estimates are NOT reliable
            end
        end
        
    otherwise
        error('Error: the selected generalized_least_sq_method for least squarres is unknown. Appropriate selections are: OLS, gfls, or gfls-AR.');
end
return
function [ LSR_params, stat_diagnostics, flag ]  = regress_prokopis( inp,out,generalized_least_sq_method,AR_lag,return_stats)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

regress_inp = inp;
regress_out = out;
%switch between different Generalized Least Squarres estimation methods
switch generalized_least_sq_method
    case 'OLS'
        mdl = fitlm(regress_inp,regress_out);                                %LSR coefficients for the inner relation.
        Cmat = mdl.Coefficients.Estimate(2:end);
        Cmat = [mean(regress_out)-mean(regress_inp)*Cmat(1:end); Cmat(1:end)];
        output_pred = [ones(length(regress_inp),1) regress_inp]*Cmat;
        
        if return_stats ==1
            %stat_diagnostics = mdl;
            
            n = length(regress_out);
            p = length(Cmat);
            
            stat_diagnostics.CoefficientCovariance  = mdl.CoefficientCovariance;
            stat_diagnostics.ParamStandardError  = mdl.Coefficients.SE;
            stat_diagnostics.ParamVals  = Cmat;
            stat_diagnostics.NumEstimatedCoefficients = p;
            stat_diagnostics.NumObservations = n;
            stat_diagnostics.DFE  = stat_diagnostics.NumObservations-stat_diagnostics.NumEstimatedCoefficients; %Degrees of freedom for errors
            stat_diagnostics.DFR  = stat_diagnostics.NumEstimatedCoefficients-1; %Degrees of freedom for regression model prediction
            stat_diagnostics.DFT  = stat_diagnostics.NumObservations-1; %Degrees of freedom for total model
            stat_diagnostics.SSE  = (norm(regress_out-output_pred))^2; %Sum of Squerres for error
            stat_diagnostics.MSE  = stat_diagnostics.SSE/(stat_diagnostics.DFE); %Mean sum of Squerres for error
            stat_diagnostics.SSR  = (norm(output_pred-mean(output_pred)))^2; %This is how matlab defines Regression Sum of Squerres (e.g. help fitglm)
            stat_diagnostics.MSR  = stat_diagnostics.SSR/stat_diagnostics.DFR; %Mean Regression Sum of Squerres
            stat_diagnostics.SST  = (norm(regress_out-mean(regress_out)))^2; %This is how matlab defines Total Sum of Squerres (e.g. help fitglm)
            stat_diagnostics.MST  = stat_diagnostics.SST/stat_diagnostics.DFT; %Mean Regression Sum of Squerres
            stat_diagnostics.NMSE = stat_diagnostics.SSE/stat_diagnostics.SST;
            for i=1:p
                stat_diagnostics.Params_tStat(i)  = stat_diagnostics.ParamVals(i)/stat_diagnostics.ParamStandardError(i);
                stat_diagnostics.Params_tPval(i) = (1-tcdf(abs(stat_diagnostics.Params_tStat(i)),stat_diagnostics.DFE))*2;
            end
            stat_diagnostics.Model_fStat  = (stat_diagnostics.SSR/stat_diagnostics.DFR)/(stat_diagnostics.SSE/stat_diagnostics.DFE);
            stat_diagnostics.Model_fPval = 1 - fcdf(stat_diagnostics.Model_fStat,stat_diagnostics.DFR,stat_diagnostics.DFE);
            [stat_diagnostics.r2,stat_diagnostics.r2adj] =rsquared(regress_out,output_pred,p);
            stat_diagnostics.loglik    = -.5*stat_diagnostics.NumObservations*(log(2*pi) + log(stat_diagnostics.SSE/stat_diagnostics.NumObservations) + 1);
            [stat_diagnostics.aic,stat_diagnostics.bic] = aicbic(stat_diagnostics.loglik,stat_diagnostics.NumEstimatedCoefficients,stat_diagnostics.NumObservations);
            %Check parameter estimation diagnostics
            if stat_diagnostics.Model_fPval<0.05
                
                %Set to 0 all params if there associated pvalue is > 0.05
                for p=1:length(Cmat)
                    if stat_diagnostics.Params_tPval(p)>0.05
                        Cmat(p)=0;
                    end
                end
                %Return only the significant coeffs LSR in Cmat.
                LSR_params = Cmat;
                flag = 0; %Used to indicated that the coeff estimates are reliable
            else
                LSR_params = zeros(size(Cmat));
                flag = 1; %Used to indicated that the coeff estimates are NOT reliable
            end
        else
            stat_diagnostics = [];
        end
        
    case 'gfls'
        
        [Cmat,se,EstCoeffCov] = fgls(regress_inp,regress_out,'intercept',false,'innovMdl','AR','arLags',AR_lag,'numIter',10);
        Cmat = [mean(regress_out)-mean(regress_inp)*Cmat(1:end); Cmat(1:end)];
        output_pred = [ones(length(regress_inp),1) regress_inp]*Cmat;
        
        
        if return_stats ==1
            n = length(regress_out);
            p = length(Cmat);
            
            stat_diagnostics.CoefficientCovariance  = EstCoeffCov;
            stat_diagnostics.ParamStandardError  = se;
            stat_diagnostics.ParamVals  = Cmat;
            stat_diagnostics.NumEstimatedCoefficients = p;
            stat_diagnostics.NumObservations = n;
            stat_diagnostics.DFE  = stat_diagnostics.NumObservations-stat_diagnostics.NumEstimatedCoefficients; %Degrees of freedom for errors
            stat_diagnostics.DFR  = stat_diagnostics.NumEstimatedCoefficients-1; %Degrees of freedom for regression model prediction
            stat_diagnostics.DFT  = stat_diagnostics.NumObservations-1; %Degrees of freedom for total model
            stat_diagnostics.SSE  = (norm(regress_out-output_pred))^2; %Sum of Squerres for error
            stat_diagnostics.MSE  = stat_diagnostics.SSE/(stat_diagnostics.DFE); %Mean sum of Squerres for error
            stat_diagnostics.SSR  = (norm(output_pred-mean(output_pred)))^2; %This is how matlab defines Regression Sum of Squerres (e.g. help fitglm)
            stat_diagnostics.MSR  = stat_diagnostics.SSR/stat_diagnostics.DFR; %Mean Regression Sum of Squerres
            stat_diagnostics.SST  = (norm(regress_out-mean(regress_out)))^2; %This is how matlab defines Total Sum of Squerres (e.g. help fitglm)
            stat_diagnostics.MST  = stat_diagnostics.SST/stat_diagnostics.DFT; %Mean Regression Sum of Squerres
            stat_diagnostics.NMSE = stat_diagnostics.SSE/stat_diagnostics.SST;
            for i=1:p
                stat_diagnostics.Params_tStat(i)  = stat_diagnostics.ParamVals(i)/stat_diagnostics.ParamStandardError(i);
                stat_diagnostics.Params_tPval(i) = (1-tcdf(abs(stat_diagnostics.Params_tStat(i)),stat_diagnostics.DFE))*2;
            end
            stat_diagnostics.Model_fStat  = (stat_diagnostics.SSR/stat_diagnostics.DFR)/(stat_diagnostics.SSE/stat_diagnostics.DFE);
            stat_diagnostics.Model_fPval = 1 - fcdf(stat_diagnostics.Model_fStat,stat_diagnostics.DFR,stat_diagnostics.DFE);
            [stat_diagnostics.r2,stat_diagnostics.r2adj] =rsquared(regress_out,output_pred,p);
            stat_diagnostics.loglik    = -.5*stat_diagnostics.NumObservations*(log(2*pi) + log(stat_diagnostics.SSE/stat_diagnostics.NumObservations) + 1);
            [stat_diagnostics.aic,stat_diagnostics.bic] = aicbic(stat_diagnostics.loglik,stat_diagnostics.NumEstimatedCoefficients,stat_diagnostics.NumObservations);
            
            %Check parameter estimation diagnostics
            if stat_diagnostics.Model_fPval<0.05
                
                %Set to 0 all params if there associated pvalue is > 0.05
                for p=1:length(Cmat)
                    if stat_diagnostics.Params_tPval(p)>0.05
                        Cmat(p)=0;
                    end
                end
                %Return only the significant coeffs LSR in Cmat.
                LSR_params = Cmat;
                flag = 0; %Used to indicated that the coeff estimates are reliable
            else
                LSR_params = zeros(size(Cmat));
                flag = 1; %Used to indicated that the coeff estimates are NOT reliable
            end
        end
        
        
    case 'gfls_AR'
        max_iter = 20;
        regress_inp_intercept = [ones(length(regress_inp),1) regress_inp];
        [Cmat, ~, se, EstCoeffCov] = gfls_AR_param_est(regress_inp_intercept,regress_out,AR_lag,max_iter);
        Cmat = [mean(regress_out)-mean(regress_inp)*Cmat(2:end); Cmat(2:end)];
        output_pred = regress_inp_intercept *Cmat;
        
        if return_stats ==1
            n = length(regress_out);
            p = length(Cmat);
            
            stat_diagnostics.CoefficientCovariance  = EstCoeffCov;
            stat_diagnostics.ParamStandardError  = se;
            stat_diagnostics.ParamVals  = Cmat;
            stat_diagnostics.NumEstimatedCoefficients = p;
            stat_diagnostics.NumObservations = n;
            stat_diagnostics.DFE  = stat_diagnostics.NumObservations-stat_diagnostics.NumEstimatedCoefficients; %Degrees of freedom for errors
            stat_diagnostics.DFR  = stat_diagnostics.NumEstimatedCoefficients-1; %Degrees of freedom for regression model prediction
            stat_diagnostics.DFT  = stat_diagnostics.NumObservations-1; %Degrees of freedom for total model
            stat_diagnostics.SSE  = (norm(regress_out-output_pred))^2; %Sum of Squerres for error
            stat_diagnostics.MSE  = stat_diagnostics.SSE/(stat_diagnostics.DFE); %Mean sum of Squerres for error
            stat_diagnostics.SSR  = (norm(output_pred-mean(output_pred)))^2; %This is how matlab defines Regression Sum of Squerres (e.g. help fitglm)
            stat_diagnostics.MSR  = stat_diagnostics.SSR/stat_diagnostics.DFR; %Mean Regression Sum of Squerres
            stat_diagnostics.SST  = (norm(regress_out-mean(regress_out)))^2; %This is how matlab defines Total Sum of Squerres (e.g. help fitglm)
            stat_diagnostics.MST  = stat_diagnostics.SST/stat_diagnostics.DFT; %Mean Regression Sum of Squerres
            stat_diagnostics.NMSE = stat_diagnostics.SSE/stat_diagnostics.SST;
            for i=1:p
                stat_diagnostics.Params_tStat(i)  = stat_diagnostics.ParamVals(i)/stat_diagnostics.ParamStandardError(i);
                stat_diagnostics.Params_tPval(i) = (1-tcdf(abs(stat_diagnostics.Params_tStat(i)),stat_diagnostics.DFE))*2;
            end
            stat_diagnostics.Model_fStat  = (stat_diagnostics.SSR/stat_diagnostics.DFR)/(stat_diagnostics.SSE/stat_diagnostics.DFE);
            stat_diagnostics.Model_fPval = 1 - fcdf(stat_diagnostics.Model_fStat,stat_diagnostics.DFR,stat_diagnostics.DFE);
            [stat_diagnostics.r2,stat_diagnostics.r2adj] =rsquared(regress_out,output_pred,p);
            stat_diagnostics.loglik    = -.5*stat_diagnostics.NumObservations*(log(2*pi) + log(stat_diagnostics.SSE/stat_diagnostics.NumObservations) + 1);
            [stat_diagnostics.aic,stat_diagnostics.bic] = aicbic(stat_diagnostics.loglik,stat_diagnostics.NumEstimatedCoefficients,stat_diagnostics.NumObservations);
            
            %Check parameter estimation diagnostics
            %             if stat_diagnostics.Model_fPval<0.05
            %
            %                 %Set to 0 all params if there associated pvalue is > 0.05
            %                 for p=1:length(Cmat)
            %                     if stat_diagnostics.Params_tPval(p)>0.05
            %                         Cmat(p)=0;
            %                     end
            %                 end
            %                 %Return only the significant coeffs LSR in Cmat.
            %                 %                 C2   =  Cmat(1:end);
            %                 %                 C2_with_intercept = [ mean(out) - mean(regress_inp,1)*C2; C2];     %Include the intercept coeff
            %                 %                 LSR_params = C2_with_intercept;
            %                 LSR_params = Cmat;
            %                 flag = 0; %Used to indicated that the coeff estimates are reliable
            %             else
            %                 stat_diagnostics = [];
            %                 LSR_params = NaN;
            %                 flag = 1; %Used to indicated that the coeff estimates are NOT reliable
            %             end
            if stat_diagnostics.Model_fPval<0.05
                
                %Set to 0 all params if there associated pvalue is > 0.05
                for p=1:length(Cmat)
                    if stat_diagnostics.Params_tPval(p)>0.05
                        Cmat(p)=0;
                    end
                end
                %Return only the significant coeffs LSR in Cmat.
                %                 C2   =  Cmat(1:end);
                %                 C2_with_intercept = [ mean(out) - mean(regress_inp,1)*C2; C2];     %Include the intercept coeff
                %                 LSR_params = C2_with_intercept;
                LSR_params = Cmat;
                flag = 0; %Used to indicated that the coeff estimates are reliable
            else
                
                %Set to 0 all params if there associated pvalue is > 0.05
                LSR_params = zeros(size(Cmat));
                flag = 1; %Used to indicated that the coeff estimates are NOT reliable
            end
        end
        
    otherwise
        error('Error: the selected generalized_least_sq_method for least squarres is unknown. Appropriate selections are: OLS, gfls, or gfls-AR.');
end
return
function [R_2,R_2_adj]=rsquared(y_orig,y_est,nparam)

if nargin<2
    error("Error: The rsquared function requires a vector of the original output measurements (y_orig), as well as a vector of the model output (y_est).")
else
    SS_res=sum( (y_orig-y_est).^2 );
    SS_tot=sum( (y_orig-mean(y_orig)).^2 );
    
    R_2=1-SS_res/SS_tot;
    
    if nargout>1
        if nargin<3
            warning('Number of model paramters is not specified. Uding default nparam=2 for caclulating R_2adj.')
            nparam=2;
        end
        R_2_adj = 1 - SS_res/SS_tot * (length(y_orig)-1)/(length(y_orig)-nparam);
    end
end
return
function [C, res_sum, Coeff_standard_err, EstCoeffCov] = gfls_AR_param_est(Vmat,out,AR_lag,max_iter)
%The script employs the Cochrane-Orcutt iterated regression algortihm to
%estimate the optimal AR lag (Feasible generalized least squares).

[nobs, nvar] = size(Vmat);
C = pinv(Vmat)*out;
resid = out - Vmat * C;

if ~AR_lag
    res_sum = norm(resid).^2;
    standard_err      = res_sum/(length(out)-length(C));
    EstCoeffCov      = standard_err*pinv(Vmat'*Vmat);
    Coeff_standard_err = sqrt(diag(EstCoeffCov));
    return
end

max_tol = min(1e-6,max(abs(C))/1000);

for r = 1:max_iter
    Beta_temp = C;
    
    X_ar = zeros(nobs-2*AR_lag,AR_lag);
    for m = 1:AR_lag
        X_ar(:,m) = resid(AR_lag+1-m:nobs-AR_lag-m);
    end
    
    Y_ar = resid(AR_lag+1:nobs-AR_lag);
    AR_para = pinv(X_ar)*Y_ar;
    
    X_main = Vmat(AR_lag+1:nobs,:);
    Y_main = out(AR_lag+1:nobs);
    for m = 1:AR_lag
        X_main = X_main-AR_para(m)*Vmat(AR_lag+1-m:nobs-m,:);
        Y_main = Y_main-AR_para(m)*out(AR_lag+1-m:nobs-m);
    end
    
    C = pinv(X_main)*Y_main;
    
    resid = out(AR_lag+1:nobs) - Vmat(AR_lag+1:nobs,:)*C;
    if max(abs(C-Beta_temp)) < max_tol
        break
    end
    
end
res_sum = norm(resid).^2;

standard_err      = res_sum/(length(out)-length(C));
EstCoeffCov      = standard_err*pinv(Vmat'*Vmat);
Coeff_standard_err = sqrt(diag(EstCoeffCov));
return
function prediction_results =  main_estimation_funct(Vmat_inp,input,out,out_sampling_rate,plsregress_options, plsregress_ncomp, plsregress_cv_folds, generalized_least_sq_method, regression_method, AR_lag, basis_set, basis_set_sampling_rate, sign_ambiguity_method)
%UNTITLED4 Summary of this function goes here
%   Estimation function for cross validation:
%
%   Input arguments:
%   inp, out:  input-output data.
%   regression_method: chooses between ordinary Least Squares REgression
%                      (LSR) or Partial Least Squarres Reression (PLSR).
%   plsregress_options: parameters for PLSR MATLAB code in case PLSR is
%                       selected
%   plsregress_ncomp: Number of PLS loadings used in regression in case
%                     PLSR is selected. plsregress_ncomp corresponds to the
%                     number of components that account most of the
%                     variance explained by PLSR. It's typically determined
%                     using k-fold cross-validation by setting the
%                     appropriate parameters in the PLSR matlab function.
%                     Do help plsregress.
%  plsregress_cv_folds: number of k-foldes in case of k-fold validation is
%                       used in PLSR to determine the number of loadings
%                       that explain most of the variance explained by the
%                       regression model.
% generalized_least_sq_method: methods used in LSR or PLSR to account for
%                              the temporal autocorrelation if the data.
%                              Availlable otions are 'OLS', 'gfls',
%                              'gfls_AR'.
%                              OLS uses an internal matlab function for
%                              ordinary least squares and no tempo.
%                              autocorr is modeled.
%                              gfls uses an internal matlab function that
%                              offers a variety of algorithms to model
%                              temporal autocorrelation.
%                              %gfls_AR assumes an AR structure for
%                              temporal autocorr. It's the fasted method
%                              that account for temp autocorr. Adapted form
%                              Wu et al.
% AR_lag: Number of data points to be used in AR modeling of temporal
%        autocorrelation is case that the "gfls_AR" is selected.
%
% Output arguments
% flag: 0 if model prediction fpval<0.01 (i.e. model prediction is more significant
%       than that achieved by the null model).
% mse:  mean squarred error evaluated on the test fold (normalized by
%       the length of the test fold.)
% nmse: normalized mse with the power of the test fold.
% bic: Bayessian information criterion evaluated using the test fold.
% aic: Akaike information criterion evaluated using the test fold
% r2:  R2 coefficient of determiantion evaluated using the test fold.
% r2adj: Adjusted R2 to acount the degrees of freedom in the regression model,
%        evaluated using the test set.
%
% REMARK: In this function, the output prediction is estimated using the
%         original input (Vmat) data and the Cmat coefficients that were estimated
%         using the outer relation (projection of PLSR coefficients back to the
%         original coefficient space). In the cross-validation estimation
%         function, the output_pred is estimated using the PLSR loadings
%         multiplied by the coeffiecients that resulted from the inner
%         relation. The MSE values obtained with the two different
%         output_pred estimates are (slightly) different. In the future it
%         would be good if output_pred is estimated in exactly the same way
%         in both functions, although that this discrepancy is not expected
%         to affect the result that much.

return_stats = 1;
switch regression_method
    case 'LSR'
        [ C, stat_diagnostics, flag ] = regress_prokopis(Vmat_inp,out,generalized_least_sq_method,AR_lag,return_stats);
    case 'PLSR'
        [ C, stat_diagnostics, flag ] = pls_prokopis(Vmat_inp,out,plsregress_ncomp,generalized_least_sq_method,AR_lag,plsregress_cv_folds,plsregress_options,return_stats);
    otherwise
        error('Error: invalid linear regression method. Valid optins are Least Squarres Regression (LSR) and Partial Least Squarres Regression (PLSR).')
end

%Use the trained params and test data to get model prediction.
Vmat_inp = [ones(size(Vmat_inp,1),1), Vmat_inp];
output_pred_tot = Vmat_inp*C;

%% Estimate output_prediction using convolution
%Find the Laguerre and input (EEG) regression parameters using SVD.
regress_params = C(2:end);
sign_ambiguity_method = 'pos_hrf_area'; %The sign method doesn't affect estimation at this point
if size(input,2)>1
    Cpar                         = reshape(regress_params,size(input,2),size(basis_set,2));
    [U_old,S,V_old]              = svd(Cpar,'econ');
    
    %Check if model prediction is not significantly different that null
    %prediction: i.e. chech is C_coeff == 0  or equivalently if S == 0.
    if isempty(find(S, 1))
        U_old = zeros(size(U_old));
        V_old = zeros(size(V_old));
    end
    %Resolve SVD sign ambiguity.
    [U_new, VS_new] = resolve_sign_ambiguity(U_old, S, V_old, Cpar, basis_set, sign_ambiguity_method);
    laguerre_regress_coeff = VS_new(:,1);
    inp_regress_coeff = U_new(:,1);
elseif size(input,2)==1
    laguerre_regress_coeff = regress_params;
    inp_regress_coeff = 1;
else
    error('Error: the number of input vectors is something weird!!! Check it out!!!');
end

k1_est   = basis_set*laguerre_regress_coeff;
% s_int = 1/basis_set_sampling_rate ;
% k1_est   = k1_est(1:s_int:end); 
inp_est  = input*inp_regress_coeff;
% full_out_pred = conv(inp_est,k1_est)+C(1);
% output_pred = full_out_pred(1:end-length(k1_est)+1);
dummy_pred =  make_convolution(k1_est, basis_set_sampling_rate, inp_est, out_sampling_rate);
output_pred_dc = mean(out-dummy_pred); 
output_pred = dummy_pred+output_pred_dc;
%% 
%Output variable
prediction_results.C_coeff_full_model     = C;
prediction_results.output_pred_dc = output_pred_dc;
prediction_results.output_pred = output_pred;
prediction_results.output_pred_tot = output_pred_tot;
prediction_results.flag        = flag;
if strcmp(regression_method,'PLSR')
prediction_results.PLSR_inner_coeff     = stat_diagnostics.ParamVals(2:end);
end
prediction_results.mse         = stat_diagnostics.MSE;
prediction_results.nmse        = stat_diagnostics.NMSE;
prediction_results.bic         = stat_diagnostics.bic;
prediction_results.aic         = stat_diagnostics.aic;
prediction_results.r2          = stat_diagnostics.r2;
prediction_results.r2adj       = stat_diagnostics.r2adj;
%prediction_results.smcF        = stat_diagnostics.smcF;
%prediction_results.smcFcritp   = stat_diagnostics.smcFcritp;

e1         = out-output_pred;
n          = length(e1);
p          = stat_diagnostics.NumEstimatedCoefficients;
testvals.NumObservations = n;
testvals.NumEstimatedCoefficients = p;
testvals.DFE   = testvals.NumObservations-testvals.NumEstimatedCoefficients; %Degrees of freedom for errors
testvals.DFR   = testvals.NumEstimatedCoefficients-1; %Degrees of freedom for regression model prediction
testvals.DFT   = testvals.NumObservations-1; %Degrees of freedom for total model
testvals.SSE   = (norm(e1))^2; %Sum of Squerres for error
testvals.MSE   = testvals.SSE/(testvals.DFE); %Mean sum of Squerres for error
testvals.NMSE  = (norm(e1))^2/(norm(out))^2;
testvals.loglik    = -.5*testvals.NumObservations*(log(2*pi) + log(testvals.SSE/testvals.NumObservations) + 1);
[testvals.aic,testvals.bic] = aicbic(testvals.loglik,testvals.NumEstimatedCoefficients,testvals.NumObservations);
return
function [ U_new, VS_new] = resolve_sign_ambiguity(U, S, V, C_params, basis_set, method)
%resolve_sign_ambiguity: this function resolves the sign ambiguity in SVD
%using four methods.
%Input variables.
% U,S,V: the products of SVD applied to C_params.
%
% C_params: the regression parameter values estimated using PLSR. These
%           values will be decomposed into the Laguerre and input (EEG)
%           regression parameters.
%
% method: the method used to resolve SVD sign ambiguity. Availlable options
%         are: 'pos_max_u_coeff', 'pos_first_nonzero_v',
%         'pos_max_v_coeff', pos_max_hrf_peak.
%
%         pos_max_u_coeff: the sign of the maximum u is positive. This
%         means that the sign of the coefficient corresponding to the input
%         that contributes the most into the regression is positive. If the
%         correlation between input/output is negative, that will result
%         into flipping of the HRF.
%
%         pos_first_nonzero_v: the sign of the first non-zero is positive.
%         pos_max_v_coeff: the sign of the maximum v is positive. This
%         means that the sign of the coefficient corresponding to the basis
%         function that contributes the most into the regression is
%         positive. If the correlation between input/output is negative,
%         that will result into flipping of the sign of the input that
%         contributes the most in explaining the output.
%
%         pos_max_hrf_peak: the hrf is estimated for each column v of
%         matrix V, and the sign of v (and of the corresponding u) is
%         fliped if the hrf peak value (at the absolutely maximum point) is
%         negative. This ensures that the HRF will always have a positive
%         peak value.
%
%         pos_hrf_area: the hrf is estimated for each column v of
%         matrix V, and the sign of v (and of the corresponding u) is
%         fliped if the hrf area is
%         negative.
%
% Remark: for all methods, the norm all columns of u is 1 and the norm of
%         each column of v is equal to the corresponding value in diag(S).

switch method
    case 'pos_max_u_coeff'
        U_new = zeros(size(U));
        V_new = zeros(size(V));
        %Search in all columns of U and flip sign if the absolute max is
        %negative.
        for i=1:length(diag(S))
            u = U(:,i);
            v = V(:,i);
            [~,ind_abs_max_U] = max(abs(u)); %Finds the maximum absolute u.
            if u(ind_abs_max_U) < 0
                u = -1*u;
                v = -1*v;
            end
            U_new(:,i) = u;
            V_new(:,i) = v;
        end
        %U_new = U_new;
        VS_new = V_new*S;
    case 'pos_first_nonzero_v'
        U_new = zeros(size(U));
        V_new = zeros(size(V));
        for i=1:length(diag(S))
            u = U(:,i);
            v = V(:,i);
            n = 1; %for first non-zero element
            ind = find(v, n,'first');
            if v(ind) < 0
                u = -1*u;
                v = -1*v;
            end
            U_new(:,i) = u;
            V_new(:,i) = v;
        end
        %U_new = U_new;
        VS_new = V_new*S;
    case 'pos_max_v_coeff'
        U_new = zeros(size(U));
        V_new = zeros(size(V));
        %Search in all columns of U and flip sign if the absolute max is
        %negative.
        for i=1:length(diag(S))
            u = U(:,i);
            v = V(:,i);
            [~,ind_abs_max_V] = max(abs(v)); %Finds the maximum absolute u.
            if v(ind_abs_max_V) < 0
                u = -1*u;
                v = -1*v;
            end
            U_new(:,i) = u;
            V_new(:,i) = v;
        end
        %U_new = U_new;
        VS_new = V_new*S;
    case 'pos_max_hrf_peak'
        U_new = zeros(size(U));
        V_new = zeros(size(V));
        %Search in all columns of V and flip sign if the absolute peak of
        % the generated HRF is negative.
        for i=1:length(diag(S))
            u = U(:,i);
            v = V(:,i);
            hrf = basis_set*v;
            %Find hrf absolute peak value
            [~, max_ind] = max(abs(hrf));
            hrf_peak_val = hrf(max_ind);
            if hrf_peak_val  < 0
                u = -1*u;
                v = -1*v;
            end
            U_new(:,i) = u;
            V_new(:,i) = v;
        end
        VS_new = V_new*S;
    case 'pos_hrf_area'
        U_new = zeros(size(U));
        V_new = zeros(size(V));
        %Search in all columns of V and flip sign if the absolute peak of
        % the generated HRF is negative.
        for i=1:length(diag(S))
            u = U(:,i);
            v = V(:,i);
            hrf = basis_set*v;
            %Find hrf area
            hrf_area = sum(hrf);
            if hrf_area  < 0
                u = -1*u;
                v = -1*v;
            end
            U_new(:,i) = u;
            V_new(:,i) = v;
        end
        VS_new = V_new*S;
end
return
%--------
function [ k_fold_cv_stats ] = k_fold_cv_conv_err(input, Vmat, output, output_sampling_rate, cv_folds, plsregress_options, plsregress_ncomp, generalized_least_sq_method, regression_method, AR_lag, basis_set, basis_set_sampling_rate )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

%Find partition (fold) indices
n = length(output);
cp   = cvpartition(n,'KFold',cv_folds);
testing_partition = cp.TestSize;

count = 0;
for i=1:length(testing_partition)
    partition_indices{i} = count+1:count+testing_partition(i);
    count = count+testing_partition(i);
end

%Find all combination of data partitions and form train and test sets
train = combnk(1:cv_folds,cv_folds-1);
all_indices = 1:cv_folds;
for i=1:size(train,1)
    Lia = ~ismember(all_indices,train(i,:));
    test(i) = all_indices(Lia);
end

%Make partitions (k-1 folds = train, 1 fold = test)
for i=1:size(train,1)
    V_mat_train_data{i} = Vmat([partition_indices{train(i,:)}],:);
    V_mat_test_data{i}  = Vmat([partition_indices{test(i)}],:);
    
    Y_mat_train_data{i} = output([partition_indices{train(i,:)}],:);
    Y_mat_test_data{i}  = output([partition_indices{test(i)}],:);
end

%Do k-fold cross validation:
%Obtain a cost value for each of the K train/test sets.
for i=1:size(train,1)
%     VTRAIN = zscore(V_mat_train_data{i}); VTEST = zscore(V_mat_test_data{i});
%     YTRAIN = zscore(Y_mat_train_data{i}); YTEST = zscore(Y_mat_test_data{i});

    VTRAIN = V_mat_train_data{i}; VTEST = V_mat_test_data{i};
    YTRAIN = Y_mat_train_data{i}; YTEST = Y_mat_test_data{i};
    make_plot =0;
    [VTRAIN, YTRAIN] = data_detrend( VTRAIN, YTRAIN, make_plot ); %Demean trainning data
    [VTEST, YTEST] = data_detrend( VTEST, YTEST, make_plot ); %Demean training data
    
    testvals =  estimation_funct_for_cross_val_conv_err(input, output, output_sampling_rate, VTRAIN,YTRAIN,VTEST,YTEST, Y_mat_test_data, plsregress_options, plsregress_ncomp, cv_folds, generalized_least_sq_method, regression_method, AR_lag, basis_set, basis_set_sampling_rate, partition_indices, test, i);
    mse(i)  = testvals.MSE;
    nmse(i) = testvals.NMSE;
end

%Get statistics for cross validation
k_fold_cv_stats.mean_mse = nanmean(mse); %Average error in each
k_fold_cv_stats.std_mse  = nanstd(mse);  %Sample mean standard deviation of cross validation MSE across folds
k_fold_cv_stats.cross_validation_standard_error = k_fold_cv_stats.std_mse/sqrt(cv_folds);
k_fold_cv_stats.mean_nmse  = nanmean(nmse);
k_fold_cv_stats.std_nmse  = nanstd(nmse);

return
function testvals =  estimation_funct_for_cross_val_conv_err(input, output, out_sampling_rate, VTRAIN,YTRAIN,VTEST,YTEST, Y_mat_test_data, plsregress_options, plsregress_ncomp, plsregress_cv_folds, generalized_least_sq_method, regression_method, AR_lag, basis_set, basis_set_sampling_rate, partition_indices, test, ind)
%UNTITLED4 Summary of this function goes here
%   Estimation function for cross validation:
%
%   Input arguments:
%   VTRAIN, YTAIN: input-output data used for training (estimate the unknown model parameters)
%   VTEST, YTEST:  input-output data used to evalluate the goodness of fit.
%   regression_method: chooses between ordinary Least Squares REgression
%                      (LSR) or Partial Least Squarres Reression (PLSR).
%   plsregress_options: parameters for PLSR MATLAB code in case PLSR is
%                       selected
%   plsregress_ncomp: Number of PLS loadings used in regression in case
%                     PLSR is selected. plsregress_ncomp corresponds to the
%                     number of components that account most of the
%                     variance explained by PLSR. It's typically determined
%                     using k-fold cross-validation by setting the
%                     appropriate parameters in the PLSR matlab function.
%                     Do help plsregress.
%  plsregress_cv_folds: number of k-foldes in case of k-fold validation is
%                       used in PLSR to determine the number of loadings
%                       that explain most of the variance explained by the
%                       regression model.
% generalized_least_sq_method: methods used in LSR or PLSR to account for
%                              the temporal autocorrelation if the data.
%                              Availlable otions are 'OLS', 'gfls',
%                              'gfls_AR'.
%                              OLS uses an internal matlab function for
%                              ordinary least squares and no tempo.
%                              autocorr is modeled.
%                              gfls uses an internal matlab function that
%                              offers a variety of algorithms to model
%                              temporal autocorrelation.
%                              %gfls_AR assumes an AR structure for
%                              temporal autocorr. It's the fasted method
%                              that account for temp autocorr. Adapted form
%                              Wu et al.
% AR_lag: Number of data points to be used in AR modeling of temporal
%        autocorrelation is case that the "gfls_AR" is selected.
%
% Output arguments
% flag: 0 if model prediction fpval<0.01 (i.e. model prediction is more significant
%       than that achieved by the null model).
% mse:  mean squarred error evaluated on the test fold (normalized by
%       the length of the test fold.)
% nmse: normalized mse with the power of the test fold.
% bic: Bayessian information criterion evaluated using the test fold.
% aic: Akaike information criterion evaluated using the test fold
% r2:  R2 coefficient of determiantion evaluated using the test fold.
% r2adj: Adjusted R2 to acount the degrees of freedom in the regression model,
%        evaluated using the test set.


return_stats = 1;
switch regression_method
    case 'LSR'
        [ C, ~, flag ] = regress_prokopis( VTRAIN,YTRAIN,generalized_least_sq_method,AR_lag,return_stats);
    case 'PLSR'
        [ C, ~, flag ] = pls_prokopis(VTRAIN,YTRAIN,plsregress_ncomp,generalized_least_sq_method,AR_lag,plsregress_cv_folds,plsregress_options,return_stats);
    otherwise
        error('Error: invalid linear regression method. Valid optins are Least Squarres Regression (LSR) and Partial Least Squarres Regression (PLSR).')
end

%Use the trained params and test data to get model prediction.
%VTEST = [ones(size(VTEST,1),1), VTEST];
%output_pred = VTEST*C;

%--------------------------------------------------------------------------
regress_params = C(2:end);
sign_ambiguity_method = 'pos_hrf_area'; %The sign method doesn't affect estimation at this point
if size(input,2)>1
    Cpar                         = reshape(regress_params,size(input,2),size(basis_set,2));
    [U_old,S,V_old]              = svd(Cpar,'econ');
    
    %Check if model prediction is not significantly different that null
    %prediction: i.e. chech is C_coeff == 0  or equivalently if S == 0.
    if isempty(find(S, 1))
        U_old = zeros(size(U_old));
        V_old = zeros(size(V_old));
    end
    %Resolve SVD sign ambiguity.
    [U_new, VS_new] = resolve_sign_ambiguity(U_old, S, V_old, Cpar, basis_set, sign_ambiguity_method);
    laguerre_regress_coeff = VS_new(:,1);
    inp_regress_coeff = U_new(:,1);
elseif size(input,2)==1
    laguerre_regress_coeff = regress_params;
    inp_regress_coeff = 1;
else
    error('Error: the number of input vectors is something weird!!! Check it out!!!');
end

k1_est   = basis_set*laguerre_regress_coeff;
% s_int = 1/basis_set_sampling_rate ;
% k1_est   = k1_est(1:s_int:end); 
inp_est  = input*inp_regress_coeff;
% full_out_pred = conv(inp_est,k1_est)+C(1);
% output_pred = full_out_pred(1:end-length(k1_est)+1);
dummy_pred =  make_convolution(k1_est, basis_set_sampling_rate, inp_est, out_sampling_rate);
output_pred_dc = mean(output-dummy_pred); 
output_pred = dummy_pred+output_pred_dc;
output_pred = output_pred([partition_indices{test(ind)}]);

%To make sure that partitions of inputs/outputs for cross valiadition
%correspond to each other, check is YTEST is same as out_test
out_test = output([partition_indices{test(ind)}],:);
make_plot =0;
[~, out_test] = data_detrend( VTRAIN, out_test, make_plot ); 
Y_mat_test_data;
if abs(sum(zscore(detrend(out_test))-zscore(detrend(YTEST))))>1e-3 
    %Subtle differences here may occure due to the detrending applied to
    %YTEST, which precedes this function. However, this differences are
    %expected to be very small. If not, it means the partition might not be
    %correct and need further investigation as to why this happens.
    error('Error: problematic partiaion');
end
%--------------------------------------------------------------------------

%Evaluation of model prerformance using test data.
e1         = YTEST-output_pred;
n          = length(e1);
p          = length(C);
%loglik     = -.5*n*(log(2*pi) + log(norm(e1)^2/n) + 1);
%[aic, bic] = aicbic(loglik,p+1,n);
[r2,r2adj] = rsquared(YTEST,output_pred,p);

%Output variable
testvals.flag  = flag;
testvals.NumObservations = n;
testvals.NumEstimatedCoefficients = p;
testvals.DFE   = testvals.NumObservations-testvals.NumEstimatedCoefficients; %Degrees of freedom for errors
testvals.DFR   = testvals.NumEstimatedCoefficients-1; %Degrees of freedom for regression model prediction
testvals.DFT   = testvals.NumObservations-1; %Degrees of freedom for total model
testvals.SSE   = (norm(e1))^2; %Sum of Squerres for error
testvals.MSE   = testvals.SSE/(testvals.DFE); %Mean sum of Squerres for error
testvals.NMSE  = (norm(e1))^2/(norm(YTEST))^2;
testvals.loglik    = -.5*testvals.NumObservations*(log(2*pi) + log(testvals.SSE/testvals.NumObservations) + 1);
[testvals.aic,testvals.bic] = aicbic(testvals.loglik,testvals.NumEstimatedCoefficients,testvals.NumObservations);
testvals.r2    = r2;
testvals.r2adj = r2adj;

return
function [model_pred_downsampled] = make_convolution(k1_est, k1_sampling_rate,inp_est, inp_sampling_rate)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[N,D] = rat(inp_sampling_rate/k1_sampling_rate);

x  = 0:2.12:2.12*(length(inp_est)-1);
xq = 0:0.1:2.12*(length(inp_est)-1);

input_q = interp1(x,inp_est,xq,'linear')';
model_pred = conv(input_q,k1_est);
model_pred_upsampled = model_pred(1:end-length(k1_est)+1);
model_pred_downsampled = resample(model_pred_upsampled,D,N);
return

