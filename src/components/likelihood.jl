#* log likleehood

module likelihood

#---- Librarys and imports -----------------------------------------------------
    using LinearAlgebra, Distributions

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
    )::Real
        
        N = length(y)
        @assert size(K_noisy, 1) == N "Dimension mismatch: y has length $N, but K has size $(size(K_noisy, 1))."
        
        # 1. Use Cholesky decomposition for stability and speed
        #   C is a colescey decomposittion object storring one triangular matrix, but multible dispatch recognices it and uses the mure clever methodes where awailable
        C = cholesky(K_noisy)
        
        # 2. Calculate the log-determinant: log|K| = logdet(C)
        log_det_K = logdet(C)
        
        # 3. Calculate the quadratic term: yᵀK⁻¹y = ||C.L \ y||²
        #   C.L returns the lower triangular Matrix for the cholesky decomposition
        quad_term = sum(abs2, C.L \ y)
        
        # 4. Sum the components
        #~   (o.p. S7 chap 3 first formular)
        log_like = -0.5 * N * log(2π) - 0.5 * log_det_K - 0.5 * quad_term
        
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
    )::Real
        
        # 1. Use Cholesky decomposition for stability and speed
        C = cholesky(K_noisy)
        
        # 2. Calculate the log-determinant
        log_det_K = logdet(C)
        
        # 3. Calculate the quadratic term
        quad_term = sum(abs2, C.L \ y)
        
        # 4. Sum the components from Equation 14 
        log_like_unnorm = - 0.5 * log_det_K - 0.5 * quad_term
        
        return log_like_unnorm
    end

end