function param = computefeature18(patch)
    window                   = fspecial('gaussian',7,7/6);
    window                   = window/sum(sum(window));
    mu                       = imfilter(patch,window,'replicate');
    mu_sq                    = mu.*mu;
    sigma                    = sqrt(abs(imfilter(patch.*patch,window,'replicate') - mu_sq));
    structdis                = (patch-mu)./(sigma+1);

    %% 2 GGD model parameters
    [shape, sigparam] = est_GGD_param(structdis(:));
    param = [shape sigparam];
    %% pair-production - AGGD - 4x4 = 16 features
    shifts = [0 1; 1 0 ; 1 1; -1 1];
    for itr_shift = 1:4
        shifted_structdis = circshift(structdis, shifts(itr_shift,:));
        pair = structdis(:) .* shifted_structdis(:);
        [alpha, leftstd, rightstd] = est_AGGD_param(pair);
        const = (sqrt(gamma(1/alpha)) / sqrt(gamma(3/alpha)));
        meanparam = (rightstd - leftstd) * (gamma(2/alpha)/gamma(1/alpha)) * const;
        param = [param alpha meanparam leftstd rightstd];               
    end
    param = param';
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

function [alpha, leftstd, rightstd] = est_AGGD_param(vec)
    gam   = 0.1:0.001:6;
    r_gam = ((gamma(2./gam)).^2)./(gamma(1./gam).*gamma(3./gam));

    leftstd            = sqrt(mean((vec(vec<0)).^2));
    rightstd           = sqrt(mean((vec(vec>0)).^2));
    gammahat           = leftstd/rightstd;
    rhat               = (mean(abs(vec)))^2/mean((vec).^2);
    rhatnorm           = (rhat*(gammahat^3 +1)*(gammahat+1))/((gammahat^2 +1)^2);
    [~, array_position] = min((r_gam - rhatnorm).^2);
    alpha              = gam(array_position);
end