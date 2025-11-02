#* building the kernel Matrix

module kernel
export build_covariance_matrix, build_covariance_matrix_ad

#---- Librarys and imports -----------------------------------------------------
    using LinearAlgebra


#---- function defenition ------------------------------------------------------
    """
        anisotropic_distance(
            x::AbstractVector{<:Real}, 
            x_prime::AbstractVector{<:Real}, 
            ϑ::AbstractVector{<:Real}
        ) -> Real

    Calculates the anisotropic distance between two data points `x` and `x_prime`
    using the relevance parameters `ϑ`. (o.p. S.3 eq. 4)

    # Arguments
    - `x`: The first data point, a d-dimensional vector.
    - `x_prime`: The second data point, a d-dimensional vector.
    - `ϑ`: The d-dimensional vector of positive relevance parameters.

    # Returns
    - A `Real` scalar representing the calculated distance.

    # Error Handling
    - Asserts that `x`, `x_prime`, and `ϑ` all have the same length.
    - Asserts that all `ϑ` values are non-negative.
    """
    function anisotropic_distance(
        x::AbstractVector{<:Real}, 
        x_prime::AbstractVector{<:Real}, 
        ϑ::AbstractVector{<:Real}
    )::Real
        
        # --- Error Handling ---
        d = length(x)
        @assert length(x_prime) == d "Input vectors `x` and `x_prime` must have the same length."
        @assert length(ϑ) == d "Input vector `x` and `ϑ` vector must have the same length."
        @assert all(>=(0.0), ϑ) "All ϑ parameters must be non-negative."
        
        # --- Implementation ---
        sum_weighted_sq_diff = 0.0
        for j ∈ 1:d
            sum_weighted_sq_diff += (x[j] - x_prime[j])^2 * ϑ[j]
        end
        
        return sqrt(sum_weighted_sq_diff)
    end

    """
        se_kernel(
            x::AbstractVector{<:Real}, 
            x_prime::AbstractVector{<:Real}, 
            ϑ::AbstractVector{<:Real}, 
            τ::Real
        ) -> Real

    Calculates the Squared Exponential (SE) kernel value between two data points
    `x` and `x_prime`. (o.p. S.3 eq. 5)

    # Arguments
    - `x`: The first data point (d-dimensional vector).
    - `x_prime`: The second data point (d-dimensional vector).
    - `ϑ`: The d-dimensional vector of relevance parameters (θ_j).
    - `τ`: The global variance parameter τ[cite: 71].

    # Returns
    - A `Real` scalar representing the kernel covariance.

    # Error Handling
    - Asserts that `τ` is positive.
    - Relies on `anisotropic_distance` for dimension checks.
    """
    function se_kernel(
        x::AbstractVector{<:Real}, 
        x_prime::AbstractVector{<:Real}, 
        ϑ::AbstractVector{<:Real}, 
        τ::Real
    )::Real
        
        # --- Error Handling ---
        @assert τ > 0.0 "The `τ` parameter (overall variance) must be positive."
        
        # --- Implementation ---
        # Note: We compute the squared distance directly to avoid the `sqrt` in
        # `anisotropic_distance` only to square it again.
        dist_sq = 0.0
        d = length(x)
        @assert length(x_prime) == d && length(ϑ) == d "Dimension mismatch between x, x_prime, and ϑ."
        @assert all(>=(0.0), ϑ) "All `theta` parameters must be non-negative"

        for j ∈ 1:d
            dist_sq += (x[j] - x_prime[j])^2 * ϑ[j]
        end
        
        return (1.0 / τ) * exp(-0.5 * dist_sq)
    end

    """
        build_covariance_matrix(
            X::Matrix{<:Real}, 
            ϑ::AbstractVector{<:Real}, 
            τ::Real; 
            σ2::Real = 0.0
        ) -> Symmetric{Float64, Matrix{Float64}}

    Constructs the full N x N covariance matrix from the input data matrix `X`
    and the kernel hyperparameters `ϑ` and `τ`.

    This function optionally adds the noise variance `σ2` to the diagonal,
    resulting in the matrix (K(x; ζ) + σ²I).

    # Arguments
    - `X`: The N x d input data matrix, where N is the number of observations
        and d is the number of features.
    - `ϑ`: The d-dimensional vector of relevance parameters.
    - `τ`: The global variance.

    # Keyword Arguments
    - `σ2`: The noise variance (σ²). This value is added *only* to the
                diagonal of the matrix. Defaults to 0.0.

    # Returns
    - A `Symmetric` N x N matrix. Using `Symmetric` is more efficient for
    downstream calculations like `cholesky` or `logdet`.

    # Error Handling
    - Asserts that the feature dimension `d` of `X` matches the length of `ϑ`.

    # Warning
    - This function should be fast and memory efficient but is not compatible with many reverse-mode AD systems
    """
    function build_covariance_matrix(
        X::Matrix{<:Real}, 
        ϑ::AbstractVector{<:Real}, 
        τ::Real; 
        σ2::Real = 0.0
    )::Symmetric{Float64, Matrix{Float64}}

        N, d = size(X)
        
        # --- Error Handling ---
        @assert length(ϑ) == d "Dimension mismatch: `X` has $d columns, but `ϑ` has length $(length(ϑ))."
        @assert σ2 >= 0.0 "Noise variance `σ2` must be non-negative."

        # --- Implementation ---
        K = Matrix{Float64}(undef, N, N)

        # We use `view` to get rows of X without allocating new memory.
        # This is more efficient than `X[i, :]` inside a loop.
        for i ∈ 1:N
            x_i = view(X, i, :)
            
            # Calculate diagonal element
            # k(x_i, x_i) is always (1/τ) since distance is 0.
            # Then add the noise variance.
            K[i, i] = (1.0 / τ) + σ2
            
            # Calculate off-diagonal elements
            for j in (i + 1):N
                x_j = view(X, j, :)
                val = se_kernel(x_i, x_j, ϑ, τ)
                K[i, j] = val
                K[j, i] = val # Exploit symmetry
            end
        end
        
        # Returning as `Symmetric` is a promise to Julia that K[i,j] == K[j,i],
        # which allows for faster and more stable linear algebra.
        return Symmetric(K)
    end

    """
        build_covariance_matrix_ad(
            X::Matrix{<:Real}, 
            ϑ::AbstractVector{<:Real}, 
            τ::Real; 
            σ2::Real = 0.0
        ) -> Symmetric{Float64, Matrix{Float64}}

    Constructs the full N x N covariance matrix in an
    AD-friendly (non-mutating, functional) way.

    This function is intended for use with reverse-mode AD systems
    like Zygote.jl, which cannot handle in-place array mutation.

    # Arguments
    - `X`: The N x d input data matrix.
    - `ϑ`: The d-dimensional vector of relevance parameters.
    - `τ`: The global variance parameter.

    # Keyword Arguments
    - `σ2`: The noise variance (σ²). This is added to the diagonal.

    # Returns
    - A `Symmetric` N x N matrix, (K + σ²I).
    """
    function build_covariance_matrix_ad(
        X::Matrix{<:Real}, 
        ϑ::AbstractVector{<:Real}, 
        τ::Real; 
        σ2::Real = 0.0
    )::Symmetric{Float64, Matrix{Float64}}

        N, d = size(X)
        
        # --- Error Handling ---
        @assert length(ϑ) == d "Dimension mismatch: `X` has $d columns, but `ϑ` has length $(length(ϑ))."
        @assert σ2 >= 0.0 "Noise variance `σ2` must be non-negative."

        # --- Implementation ---
        
        # 1. Create a vector of "views" for each row.
        #    `view` is non-allocating and AD-friendly.
        X_rows = [view(X, i, :) for i in 1:N]

        # 2. Use a matrix comprehension to build K.
        #    This is pure and functional. It creates a new matrix `K`
        #    by calling `se_kernel` for every (i, j) pair.
        K = [se_kernel(X_rows[i], X_rows[j], ϑ, τ) for i in 1:N, j in 1:N]

        # 3. Add the diagonal noise.
        #    Creating the Identity matrix `I` and adding it is also
        #    a pure, AD-friendly operation.
        K_noisy = K + (σ2 * I(N))
        
        # 4. Return as Symmetric.
        #    Symmetric(.) is also differentiable.
        return Symmetric(K_noisy)
    end


end