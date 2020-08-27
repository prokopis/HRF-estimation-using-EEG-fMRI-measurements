input = randn(5000,4);
eeg_coeff = [0.4264, 0.2132, 0.8528, 0.2132];
alpha = 0.7;
mu = 3.5;
sigma=1; 
num_of_basis_functs=2;
fMRI_sampling_rate=2.12; 
kernel_length=35;
bf_sampling_rate = 0.1;

[basis,basis_sampling_rate] = laguerre_basis_smothed_with_gaussian_kernel( alpha, mu, sigma, num_of_basis_functs, bf_sampling_rate, kernel_length );
[N,D] = rat(fMRI_sampling_rate/basis_sampling_rate);
hrf = 0.6*basis(:,1)+0.4*basis(:,2);
hrf_fmri_tr = resample(hrf,D,N);

input_tot = input*eeg_coeff';

%Generate BOLD timeseries as the convolution of the input timeseries with
%the HRF.
BOLD = conv(input_tot, hrf_fmri_tr);
BOLD = BOLD(1:end-length(hrf_fmri_tr)-1,:);

save('input', 'input_tot','eeg_coeff', 'BOLD', 'hrf');


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

end
function [L_out] = generalized_laguerre(a,p,r)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

out = 0;
for j=0:p
    out = out + nchoosek(p+a,p-j).*((-r)^j)./factorial(j);
end
L_out = out;

end
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
end
