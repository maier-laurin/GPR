#* prior (log() denseties
    # we are not implementing them according to Julias Distributions.jl Interface, because we don't need/want the normilized versions
    # which these interface would require, but that is not a problem because we will implement everything 
    # that would require this interface ourself

#---- Librarys and imports -----------------------------------------------------
    using SpecialFunctions # Gamma function for `kummer_U`
    using HypergeometricFunctions # Parts for `kummer_U`

#---- additionall implementation -----------------------------------------------

    #TODO check for correctness or build something different 
    #! this is just the result of a 2 hour discussion with a large language model about 
    #! if theres a implementation of the confluent hypergeometric function of the second
    #! kind in julia that does not call C++ over GLS because that screams trouble with AD. 
    #* And this is the best we came up with
    """
        kummer_U(a, b, z)

    Calculates the confluent hypergeometric function of the second kind, U(a, b, z),
    using its definition in terms of the function of the first kind, ₁F₁ (Kummer's M).
    """
    function kummer_U(a::Number, b::Number, z::Number)
        # The identity requires two terms
        
        # Term 1: (Γ(1-b) / Γ(a-b+1)) * ₁F₁(a; b; z)
        term1 = (gamma(1 - b) / gamma(a - b + 1)) * _₁F₁(a, b, z)
        
        # Term 2: (Γ(b-1) / Γ(a)) * z^(1-b) * ₁F₁(a-b+1; 2-b; z)
        term2 = (gamma(b - 1) / gamma(a)) * z^(1 - b) * _₁F₁(a - b + 1, 2 - b, z)
        
        return term1 + term2
    end


#---- function defenition ------------------------------------------------------

    """
        log_prior_sigma2_unnorm(σ²::Real; λ::Real = 10.0) -> Real

    Calculates the unnormalized log-prior for the noise variance `σ²`. (o.p. S.13 eq. 17)

    The prior is an Exponential distribution (o.p. S.6 eq. 10). We interpret this
    as a rate parameter `λ = 10`.

    The unnormalized log-prior is: -λ * σ²

    # Arguments
    - `σ²`: The noise variance (must be non-negative).

    # Keyword Arguments
    - `λ`: The rate parameter of the Exponential prior. Defaults to 10.0.

    # Returns
    - A `Real` scalar representing the unnormalized log-density.
    """
    function log_prior_sigma2_unnorm(σ²::Real; λ::Real = 10.0)::Real
        @assert σ² >= 0.0 "Noise variance σ² must be non-negative."
        @assert λ > 0.0 "Rate parameter λ must be positive."
        
        return -λ * σ²
    end

    """
        log_prior_tau_unnorm(τ::Real, a::Real, c::Real) -> Real

    Calculates the unnormalized log-prior for the global shrinkage
    parameter `τ` (o.p. S.13 eq. 16), which follows an F-distribution (o.p. S.5 eq. 8).

    # Arguments
    - `τ`: The global shrinkage parameter (must be positive).
    - `a`: The first hyperparameter (must be positive).
    - `c`: The second hyperparameter (must be positive).

    # Returns
    - A `Real` scalar representing the unnormalized log-density.
    """
    function log_prior_tau_unnorm(τ::Real, a::Real, c::Real)::Real
        @assert τ > 0.0 "Global shrinkage parameter τ must be positive."
        @assert a > 0.0 "Hyperparameter a must be positive."
        @assert c > 0.0 "Hyperparameter c must be positive."

        return (c - 1) * log(τ) - (c + a) * log(1.0 + (c / a) * τ)
    end

    """
        log_prior_triple_gamma_unnorm(ϑ_j::Real, τ::Real, a::Real, c::Real) -> Real

    Calculates the unnormalized log-prior for a *single* relevance
    parameter `ϑ_j` from the triple gamma distribution (marginalized).

    This implements the term inside the sum in Equation (o.p. S.13 eq. 15).

    # Arguments
    - `ϑ_j`: A single relevance parameter (must be positive).
    - `τ`: The global shrinkage parameter (must be positive).
    - `a`: The first hyperparameter (must be positive, and `1.5 - a`
        must be valid for `kummer_U`).
    - `c`: The second hyperparameter (must be positive).

    # Returns
    - A `Real` scalar representing the unnormalized log-density.
    """
    function log_prior_triple_gamma_unnorm(ϑ_j::Real, τ::Real, a::Real, c::Real)::Real
        # --- Error Handling ---
        @assert ϑ_j > 0.0 "Relevance parameter ϑ_j must be positive."
        @assert τ > 0.0 "Global shrinkage parameter τ must be positive."
        @assert a > 0.0 "Hyperparameter a must be positive."
        @assert c > 0.0 "Hyperparameter c must be positive."
        # we try to keep the entire function type stable, operations that contain untyped constants promote automatically to float64
        T = float(promote_type(typeof(ϑ_j), typeof(τ), typeof(a), typeof(c)))
        # --- Implementation ---
        κ = (T(2.0) * c) / (τ * a)
        # Calculate arguments for kummer_U
        arg_a = c + T(0.5)
        arg_b = T(1.5) - a
        arg_z = ϑ_j / (T(2) * κ)
        
        @assert arg_z > 0.0 "Argument `z` for kummer_U must be positive."

        # 3. Calculate log(U(...))
        #    We use log(kummer_U(...)) for numerical stability.
        #    We add the floating point epsilon at 0 to the output in case kummer_U underflows to 0
        #       this does not really add that much of a possitive bias because in all cases that arn't 0.0 + eps the eps gets 
        #       swallowed in the floating point precission and the result is like it was never added in the first place
        log_U = log(kummer_U(arg_a, arg_b, arg_z) + nextfloat(zero(typeof(arg_z))))
        
        # 4. Calculate log-density from Equation 15
        log_dens = T(0.5) * (log(τ) - log(ϑ_j)) + log_U
        
        return log_dens
    end


    """
        log_prior_unnorm(
            ϑ::AbstractVector{<:Real}, 
            τ::Real, 
            σ2::Real; 
            a::Real, 
            c::Real, 
            λ::Real = 10.0
        ) -> Real

    Calculates the *total* unnormalized log-prior for all parameters.

    This function sums the log-priors for ϑ (all components), τ, and σ².

    # Arguments
    - `ϑ`: A d-dimensional vector of relevance parameters.
    - `τ`: The global shrinkage parameter.
    - `σ²`: The noise variance.

    # Keyword Arguments
    - `a`: The first hyperparameter for the triple gamma and F-priors.
    - `c`: The second hyperparameter for the triple gamma and F-priors.
    - `λ`: The rate parameter for the σ² prior. Defaults to 10.0.

    # Returns
    - A `Real` scalar representing the total unnormalized joint log-prior.
    """
    function log_prior_unnorm(
        ϑ::AbstractVector{<:Real}, 
        τ::Real, 
        σ²::Real; 
        a::Real, 
        c::Real, 
        λ::Real = 10.0
    )::Real

        # 1. Sum over all triple gamma components
        log_p_ϑ = 0.0
        for ϑ_j in ϑ
            log_p_ϑ += log_prior_triple_gamma_unnorm(ϑ_j, τ, a, c)
        end
        
        # 2. Calculate log-prior for τ
        log_p_τ = log_prior_tau_unnorm(τ, a, c)
        
        # 3. Calculate log-prior for σ²
        log_p_σ² = log_prior_sigma2_unnorm(σ², λ=λ)
        
        # 4. Return total joint log-prior
        return log_p_ϑ + log_p_τ + log_p_σ²
    end

#---- Testing ------------------------------------------------------------------
    #=
    # 1. Define hyperparameters
    a_val = 0.1
    c_val = 0.1 
    λ_val = 10.0 


    using Plots

    # 2. Define dummy parameters
    ϑ_params = [0.2, 1.5, 0.01]
    τ_param  = 0.5
    σ²_param = 0.1

    # 3. Test individual components
    lp_tg = log_prior_triple_gamma_unnorm(ϑ_params[1], τ_param, a_val, c_val)
    plot(x -> log_prior_triple_gamma_unnorm(ϑ_params[1], log_prior_triple_gamma_unnorm(ϑ_params[1],x, a_val, c_val), a_val, c_val), 0.01, 2)

    lp_f = log_prior_tau_unnorm(τ_param, a_val, c_val)
    plot(x -> log_prior_tau_unnorm(x, a_val, c_val), 0.01, 2)

    lp_exp = log_prior_sigma2_unnorm(σ²_param, λ=λ_val)
    plot(x -> log_prior_sigma2_unnorm(x, λ=λ_val), 0.01, 2)

    # 4. Test the total joint log-prior
    total_log_prior = log_prior_unnorm(
        ϑ_params, τ_param, σ²_param, 
        a=a_val, c=c_val, λ=λ_val
    )
    plot(x -> log_prior_unnorm(
        ϑ_params, x, σ²_param, 
        a=a_val, c=c_val, λ=λ_val
    ), 0.01, 2)
    =#