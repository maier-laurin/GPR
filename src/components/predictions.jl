#*

module Predict
export predictive_distribution, predictive_distribution_marginal

#---- Librarys and imports -----------------------------------------------------
    using Distributions
    using LinearAlgebra
    using ..Kernels
#---- functions ----------------------------------------------------------------
    #* prdictive distribution for fixed parameter

    """
        predictive_distribution(
            X_test::Matrix{<:Real}, 
            X_train::Matrix{<:Real}, 
            y_train::AbstractVector{<:Real},
            ϑ::AbstractVector{<:Real},
            τ::Real,
            σ2::Real
        ) -> MvNormal

    Calculates the posterior predictive distribution for new data points `X_test`,
    given training data `(X_train, y_train)` and a fixed set of
    hyperparameters {ϑ, τ, σ²}.

    Returns a `MvNormal` object.

    *but is numerically unstable for prediction points close to each other* i.e the resulting Kovarianz matrix is not PD, therfore MVNormal cancels with an error.
    """
    function predictive_distribution(
        X_test::Matrix{<:Real}, 
        X_train::Matrix{<:Real}, 
        y_train::AbstractVector{<:Real},
        ϑ::AbstractVector{<:Real},
        τ::Real,
        σ2::Real
    )::MvNormal
        
        # Dimension and Parameter Checking
        N_train, d_train = size(X_train)
        N_test, d_test = size(X_test)
        
        @assert d_train == d_test "Feature dimensions mismatch: X_train has $d_train features, X_test has $d_test."
        @assert length(ϑ) == d_train "Hyperparameter mismatch: `ϑ` has length $(length(ϑ)), but data has $d_train features."
        @assert length(y_train) == N_train "Data mismatch: `X_train` has $N_train rows, but `y_train` has $(length(y_train)) elements."
        @assert τ > 0.0 "Hyperparameter `τ` must be positive."
        @assert σ2 >= 0.0 "Hyperparameter `σ²` must be non-negative."
        @assert all(>=(0.0), ϑ) "All relevance parameters `ϑ_j` must be non-negative."

        # Build the necessary kernel matrices
        
        # K(X, X) + σ²I
        K_y = build_covariance_matrix(X_train, ϑ, τ, σ2=σ2)
        
        # K(X*, X*)
        K_star_star = build_covariance_matrix(X_test, ϑ, τ, σ2=0.0)
        
        # K(X, X*)
        K_star = Matrix{Float64}(undef, N_train, N_test)
        X_train_rows = [view(X_train, i, :) for i in 1:N_train]
        X_test_rows = [view(X_test, i, :) for i in 1:N_test]
        
        for i in 1:N_train
            for j in 1:N_test
                K_star[i, j] = se_kernel(X_train_rows[i], X_test_rows[j], ϑ, τ)
            end
        end

        # Compute Posterior Mean (μ_*)
        C_y = cholesky(K_y)
        α = C_y \ y_train 
        μ_star = K_star' * α
        
        # Compute Posterior Covariance (Σ_*)
        V = C_y \ K_star

        #! probably the cause for the numeric instability
        Σ_f_star = Symmetric(K_star_star - V' * V)
        
        Σ_y_star = Symmetric(Σ_f_star + σ2 * I)
        print(Σ_y_star)
        jitter = 1e-1
        Σ_final = Symmetric(Σ_y_star + jitter * I)
        return MvNormal(μ_star, Σ_final)
    end

    """
        predictive_distribution_marginal(X_test, X_train, y_train, ϑ, τ, σ2) -> Vector{Normal}

    Calculates the posterior predictive distribution of a Gaussian Process, computing
    only the *marginal* Normal distribution for each test point.

    This function is designed for numerical stability and memory efficiency. By
    calculating the marginal predictive distribution for each test point separately, it avoids
    the construction of the full predictive covariance matrix,
    which is prone to numerical instability (causing `PosDefException`) and uses
    O(N²) memory.

    This approach is ideal for plotting with the `plot_gp_heatmap` function. However it can't be used to draw full, correlated samples from the
    posterior.

    # Arguments
    - `X_test::Matrix{<:Real}`: The test input data (size `N_test x d`).
    - `X_train::Matrix{<:Real}`: The training input data (size `N_train x d`).
    - `y_train::AbstractVector{<:Real}`: The training output data (length `N_train`).
    - `ϑ::AbstractVector{<:Real}`: Vector of kernel relevance parameters.
    - `τ::Real`: The kernel scaling hyperparameter.
    - `σ2::Real`: The observation noise variance.

    # Returns
    - A Vector of univariate Normal Distributions
    """

    function predictive_distribution_marginal(
        X_test::Matrix{<:Real}, 
        X_train::Matrix{<:Real}, 
        y_train::AbstractVector{<:Real},
        ϑ::AbstractVector{<:Real},
        τ::Real,
        σ2::Real
    )
        
        # Dimension and Parameter Checking
        N_train, d_train = size(X_train)
        N_test, d_test = size(X_test)
        
        @assert d_train == d_test "Feature dimensions mismatch: X_train has $d_train features, X_test has $d_test."
        @assert length(ϑ) == d_train "Hyperparameter mismatch: `ϑ` has length $(length(ϑ)), but data has $d_train features."
        @assert length(y_train) == N_train "Data mismatch: `X_train` has $N_train rows, but `y_train` has $(length(y_train)) elements."
        @assert τ > 0.0 "Hyperparameter `τ` must be positive."
        @assert σ2 >= 0.0 "Hyperparameter `σ²` must be non-negative."
        @assert all(>=(0.0), ϑ) "All relevance parameters `ϑ_j` must be non-negative."
        
        T = promote_type(eltype(X_test), eltype(X_train), eltype(y_train), eltype(ϑ), typeof(τ), typeof(σ2))

        #§ μ* = K(X*, X)(K(X, X) + σ2 I)^-1 y
        # for efficency reasins we will calculate the vector (K(X, X) + σ2 I)^-1 y first outside the loop
        # Compute Cholesky factor

        K_y = build_covariance_matrix(X_train, ϑ, τ, σ2=σ2)
        C_y = cholesky(Symmetric(K_y))
        
        # 2. Compute α vector
        α = C_y \ y_train 
        
        # Allocate output vector
        N_star = Vector{Normal}(undef, N_test)

        # Loop over test points
        Threads.@threads for j ∈ 1:N_test
            # Get the j-th test point
            x_j = view(X_test, j, :)
            
            # Build the cross-covariance vector k* = K(X_train, x_j)
            k_star_j = Vector{Float64}(undef, N_train)
            for i in 1:N_train
                k_star_j[i] = se_kernel(view(X_train, i, :), x_j, ϑ, τ)
            end
            
            # "self-covariance" scalar
            k_star_star_j = se_kernel(x_j, x_j, ϑ, τ)

            # mean
            μ_star = dot(k_star_j, α)
            
            # variance
            V_j = C_y.L \ k_star_j
            
            # Σ_f_star[j,j] = k(x_j, x_j) - V_j' * V_j
            var_f_j = k_star_star_j - dot(V_j, V_j)
            
            #§ maby var_f_j might be some machine epsilons smaller than 0 because of floating point rounding errors
            σ2_star_diag = max(T(0.0), var_f_j) + σ2

            N_star[j] = Normal(μ_star, σ2_star_diag)
        end
        
        return N_star
    end
end