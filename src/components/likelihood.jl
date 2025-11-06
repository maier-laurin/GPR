#* log likleehood

module Likelihood
export gpr_log_likelihood, gpr_log_likelihood_unnorm

#---- Librarys and imports -----------------------------------------------------
    using LinearAlgebra
    using Distributions

#---- function defenition ------------------------------------------------------
    """
        gpr_log_likelihood(
            y::AbstractVector{<:Real}, 
            K_noisy::Symmetric{<:Real}
        ) -> Real

    Calculates the marginalized Gaussian Process log-likelihood (o.p. S.6 eq. 10).

    Uses a numerically stable Cholesky decomposition.

    # Arguments
    - `y`: The N-dimensional observation vector.
    - `K_noisy`: The N x N *noisy* covariance matrix (K + σ²I). 
                Must be `Symmetric` and positive definite.

    # Returns
    - A `Real` scalar representing the log-likelihood.
    """
    function gpr_log_likelihood(
        y::AbstractVector{<:Real}, 
        K_noisy::Symmetric{<:Real}
    )

        T = promote_type(eltype(y), eltype(K_noisy))
        
        N = length(y)
        @assert size(K_noisy, 1) == N "Dimension mismatch: y has length $N, but K has size $(size(K_noisy, 1))."
        
        # Use Cholesky decomposition for stability and speed
        #   C is a colescey decomposittion object storring one triangular matrix, but multible dispatch recognices it and uses the mure clever methodes where awailable
        C = cholesky(K_noisy)
        
        # Calculate the log-determinant: log|K| = logdet(C)
        log_det_K = logdet(C)
        
        # 3. Calculate the quadratic term: yᵀK⁻¹y = ||C.L \ y||²
        #   C.L returns the lower triangular Matrix for the cholesky decomposition
        quad_term = sum(abs2, C.L \ y)
        
        # 4. Sum the components
        #~   (o.p. S7 chap 3 first formular)
        log_like = -T(0.5) * N * log(T(2π)) - T(0.5) * log_det_K - T(0.5) * quad_term
        
        return log_like
    end

    """
        gpr_log_likelihood_unnorm(
            y::AbstractVector{<:Real}, 
            K_noisy::Symmetric{<:Real}
        ) -> Real

    Calculates the *unnormalized* marginalized Gaussian Process log-likelihood(o.p. S.13 eq. 14).

    This function is identical to `gpr_log_likelihood` but omits the
    Gaussian normalization constant `(-N/2 * log(2π))`.

    log p(y|K) ∝ -1/2 * log|K| - 1/2 * yᵀK⁻¹y

    # Arguments
    - `y`: The N-dimensional observation vector.
    - `K_noisy`: The N x N *noisy* covariance matrix (K + σ²I). 
                Must be `Symmetric` and positive definite.

    # Returns
    - A `Real` scalar representing the unnormalized log-likelihood.
    """
    function gpr_log_likelihood_unnorm(
        y::AbstractVector{<:Real}, 
        K_noisy::Symmetric{<:Real}
    )

        T = promote_type(eltype(y), eltype(K_noisy))
        
        # Use Cholesky decomposition for stability and speed
        C = cholesky(K_noisy)
        
        # Calculate the log-determinant
        log_det_K = logdet(C)
        
        # Calculate the quadratic term
        quad_term = sum(abs2, C.L \ y)
        
        # Sum the components from Equation 14 
        log_like_unnorm = - T(0.5) * log_det_K - (0.5) * quad_term
        
        return log_like_unnorm
    end

end