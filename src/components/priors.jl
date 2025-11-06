#* prior (log() denseties
    # we are not implementing them according to Julias Distributions.jl Interface, because we don't need/want the normilized versions
    # which these interface would require, but that is not a problem because we will implement everything 
    # that would require this interface ourself

module Priors
export log_prior_unnorm

#---- imports ------------------------------------------------------------------
    using ..SpecialSpecialFunctions
#---- function defenition ------------------------------------------------------

    """
        log_prior_sigma2_unnorm(σ²::Real; λ::Real = 10.0) -> Real

    Calculates the unnormalized log-prior for the noise variance `σ²`. (o.p. S.13 eq. 17)

    The prior is an Exponential distribution (o.p. S.6 eq. 10). We interpret this
    as a rate parameter `λ = 10`.

    The unnormalized log-prior is: -λ * σ²

    # Arguments
    - `σ2`: The noise variance (must be non-negative).

    # Keyword Arguments
    - `λ`: The rate parameter of the Exponential prior. Defaults to 10.0.
    # Returns
    - A `Real` scalar representing the unnormalized log-density.
    """
    function log_prior_sigma2_unnorm(σ2::T; λ::Real = T(10.0)) where {T <: Real}
        @assert σ2 >= 0.0 "Noise variance σ² must be non-negative."
        @assert λ > 0.0 "Rate parameter λ must be positive."
        
        return -λ * σ2
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
    function log_prior_tau_unnorm(τ::Real, a::Real, c::Real)
        @assert τ > 0.0 "Global shrinkage parameter τ must be positive."
        @assert a > 0.0 "Hyperparameter a must be positive."
        @assert c > 0.0 "Hyperparameter c must be positive."

        T = promote_type(typeof(τ), typeof(a), typeof(c))

        return (c - T(1)) * log(τ) - (c + a) * log(T(1) + (c / a) * τ)
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
    function log_prior_triple_gamma_unnorm(ϑ_j::Real, τ::Real, a::Real, c::Real)
        # --- Error Handling ---
        @assert ϑ_j > 0.0 "Relevance parameter ϑ_j must be positive."
        @assert τ > 0.0 "Global shrinkage parameter τ must be positive."
        @assert a > 0.0 "Hyperparameter a must be positive."
        @assert c > 0.0 "Hyperparameter c must be positive."
        # we try to keep the entire function type stable, operations that contain untyped constants promote automatically to float64
        T = promote_type(typeof(ϑ_j), typeof(τ), typeof(a), typeof(c))
        
        # --- Implementation ---
        κ = (T(2) * c) / (τ * a)
        # Calculate arguments for kummer_U
        arg_a = c + T(0.5)
        arg_b = T(1.5) - a
        arg_z = ϑ_j / (T(2) * κ)
        
        @assert arg_z > 0.0 "Argument `z` for kummer_U must be positive."

        # Calculate log(U(...))
        #    We use log(kummer_U(...)) for numerical stability.
        #    We add the floating point epsilon at 0 to the output in case kummer_U "underflows" to 0
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
        σ2::T1; 
        a::Real, 
        c::Real, 
        λ::Real = T1(10)
    ) where {T1 <: Real}

        T = promote_type(eltype(ϑ), typeof(τ), typeof(a), typeof(σ2), typeof(c), typeof(λ))
        # Sum over all triple gamma components
        log_p_ϑ = zero(T)
        for ϑ_j in ϑ
            log_p_ϑ += log_prior_triple_gamma_unnorm(ϑ_j, τ, a, c)
        end
        
        # Calculate log-prior for τ
        log_p_τ = log_prior_tau_unnorm(τ, a, c)
        
        # 3. Calculate log-prior for σ²
        log_p_σ2 = log_prior_sigma2_unnorm(σ2, λ=λ)
        
        # 4. Return total joint log-prior
        return log_p_ϑ + log_p_τ + log_p_σ2
    end

end