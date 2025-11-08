#* Unnormaliced log posterior
module Posteriors
export total_log_posterior_unnorm
    
#---- Librarys and imports -----------------------------------------------------
    using LinearAlgebra
    using Distributions
    using ..Kernels
    using ..Priors
    using ..Likelihood

#---- function defenition -----------------------------------------------------
    """
        total_log_posterior_unnorm(
            p_log::AbstractVector{<:Real},
            X::Matrix{<:Real},
            y::AbstractVector{<:Real};
            a::Real,
            c::Real
        ) -> Real

    Calculates the total unnormalized log-posterior of a Gaussian process,
    given the logariced parameters, at a set of samples.
    This function will be the target for the Bayesian methods.

    This function is AD-friendly and suitable for maximization.

    It computes:
    log p(ξ|y) = log p(y|ξ) + log p(ξ) (o.p. S.13)

    # Arguments
    - `p_log`: A single (d+2)-dimensional vector of *unconstrained*
            parameters in log-space: [log(ϑ₁)...log(ϑ_d), log(τ), log(σ²)].
    - `X`: The N x d input data matrix.
    - `y`: The N-dimensional observation vector.

    # Keyword Arguments
    - `a`: The first hyperparameter for the triple gamma and F-priors.
    - `c`: The second hyperparameter for the triple gamma and F-priors.

    # Returns
    - A `Real` scalar. Returns `-Inf` if the parameters are invalid
    (e.g., leading to a non-positive-definite matrix).
    """
    function total_log_posterior_unnorm(
        p_log::AbstractVector{<:Real},
        X::Matrix{<:Real},
        y::AbstractVector{<:Real};
        a::Real,
        c::Real
    )

        N, d = size(X)
        @assert length(p_log) == d + 2 "Parameter vector `p_log` has wrong length."

        try
            # Reparameterization (Unconstrained -> Constrained)
            ϑ = exp.(p_log[1:d])
            τ = exp(p_log[d+1])
            σ² = exp(p_log[d+2])

            # Build AD-Friendly Covariance Matrix
            K_noisy = build_covariance_matrix_ad(X, ϑ, τ, σ2=σ²)
            
            # Calculate Unnormalized Log-Likelihood
            log_like = gpr_log_likelihood_unnorm(y, K_noisy)
            
            # Calculate Unnormalized Log-Prior
            log_prior = log_prior_unnorm(ϑ, τ, σ²; a=a, c=c)
            
            # Return Total Log-Posterior
            return log_like + log_prior

        catch e
            # If Cholesky fails (PosDefException) or params are bad (DomainError),
            # return -Inf. This tells any optimizer/sampler that this is an
            # invalid region of the parameter space.
            #§ at least thats the thory prbably will this Inf approach cause problems later on
            if isa(e, PosDefException) || isa(e, DomainError)
                return -Inf
            else
                rethrow(e)
            end
        end
    end

end