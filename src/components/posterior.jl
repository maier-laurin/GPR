#* Unnormaliced log posterior

#---- Librarys and imports -----------------------------------------------------
    using LinearAlgebra
    using Distributions
    include("kernel_matrix.jl")
    include("prior_distribution.jl")
    include("GPR_log_liklihood.jl")

#---- function defenition -----------------------------------------------------
    """
        total_log_posterior_unnorm(
            p_log::AbstractVector{<:Real},
            X::Matrix{<:Real},
            y::AbstractVector{<:Real};
            a::Real,
            c::Real
        ) -> Real

    Calculates the total unnormalized log-posterior density, the target
    for the Bayesian methods.

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
    )::Real

        N, d = size(X)
        @assert length(p_log) == d + 2 "Parameter vector `p_log` has wrong length."

        try
            # --- 1. Reparameterization (Unconstrained -> Constrained) ---
            ϑ = exp.(p_log[1:d])
            τ = exp(p_log[d+1])
            σ² = exp(p_log[d+2])

            # --- 2. Build AD-Friendly Covariance Matrix ---
            K_noisy = build_covariance_matrix_ad(X, ϑ, τ, σ2=σ²)
            
            # --- 3. Calculate Unnormalized Log-Likelihood ---
            log_like = gpr_log_likelihood_unnorm(y, K_noisy)
            
            # --- 4. Calculate Unnormalized Log-Prior ---
            log_prior = log_prior_unnorm(ϑ, τ, σ²; a=a, c=c)
            
            # --- 5. Return Total Log-Posterior ---
            return log_like + log_prior

        catch e
            # If Cholesky fails (PosDefException) or params are bad (DomainError),
            # return -Inf. This tells any optimizer/sampler that this is an
            # invalid region of the parameter space.
            if isa(e, PosDefException) || isa(e, DomainError)
                return -Inf
            else
                rethrow(e)
            end
        end
    end

#---- Testing ------------------------------------------------------------------

    # 1. Get the optimized parameters from the ML model
        N_obs = 100
        N_features = 3
        X_data = randn(N_obs, N_features)
        y_data = sin.(X_data[:, 1]*2) .* 0.5 + randn(N_obs) .* 0.1
        include("maximum_likleehood_Model.jl")
        ml_result = train_ml_model(X_data, y_data, num_restarts=10)

        p_log_optimized = ml_result.params_log

    # 2. Define the prior settings
        a_hs = 0.500001 #? why is does it evaluate to Inf for a = 0.5?
        c_hs = 0.5

        a_tg = 0.1
        c_tg = 0.1

    # 3. Calculate the total log-posterior
        log_post_hs = total_log_posterior_unnorm(
            p_log_optimized, X_data, y_data, a=a_hs, c=c_hs
        )

        log_post_tg = total_log_posterior_unnorm(
            p_log_optimized, X_data, y_data, a=a_tg, c=c_tg
        )

    println("Log-Posterior (Horseshoe): ", log_post_hs)
    println("Log-Posterior (Triple Gamma): ", log_post_tg)