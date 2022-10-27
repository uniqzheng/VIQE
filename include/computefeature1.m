function param = computefeature1(patch)
    window                   = fspecial('gaussian',7,7/6);
    window                   = window/sum(sum(window));
    mu                       = imfilter(patch,window,'replicate');
    mu_sq                    = mu.*mu;
    sigma                    = sqrt(abs(imfilter(patch.*patch,window,'replicate') - mu_sq));
    structdis                = (patch-mu)./(sigma+1);

    %% 2 GGD model parameters
    [shape, sigparam] = est_GGD_param(structdis(:));
    param = [shape;sigparam];
end
function [beta_par, alpha_par] = est_GGD_param(vec)
    % moment matching to estimate \beta
    gam                              = 0.1:0.001:6;
    r_gam                            = (gamma(1./gam).*gamma(3./gam))./((gamma(2./gam)).^2);
    sigma_sq                         = mean((vec).^2);
    alpha_par                            = sqrt(sigma_sq);
    E                                = mean(abs(vec));
    rho                              = sigma_sq/E^2;
    [~, array_position] = min(abs(rho - r_gam));
    beta_par                         = gam(array_position);
    % close form equation of entropy
    % entr_par = (1/beta_par) - log(beta_par/(2*alpha_par*gamma(1/beta_par)));
    % close form equation of kurtosis
    % kurt_par = gamma(5/beta_par)*gamma(1/beta_par)/(gamma(3/beta_par)^2);
end