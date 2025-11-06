#* data_generation

module DataGeneration
export generate_gp_data

#---- Librarys and imports -----------------------------------------------------
    using LinearAlgebra
    using Distributions
    using ..Kernels
    
#---- functions ----------------------------------------------------------------
    """
        generate_gp_data(
            X::Matrix{<:Real},
            ϑ::AbstractVector{<:Real},
            τ::Real,
            σ2::Real
        ) -> Vector{Float64}

    Generates a realisation of a GP with a SE kernel given ϑ, τ and σ² at the points in X

    y ~ N(0, K(x; ϑ, τ) + σ²I)

    # Arguments
    - `X`: The N x d input data matrix.
    - `ϑ`: The d-dimensional vector of *true* relevance parameters.
    - `τ`: The *true* global variance parameter.
    - `σ²`: The *true* noise variance (paper uses 0.1 ).

    # Returns
    - A `Vector{Float64}` of length N, representing the sampled `y` vector.

    # Error Handling
    - Asserts all dimensional compatibilities.
    - Relies on `build_covariance_matrix` for hyperparameter checks.
    """
    function generate_gp_data(
        X::Matrix{<:Real},
        ϑ::AbstractVector{<:Real},
        τ::Real,
        σ2::Real
    )::Vector{Float64}
        
        N, d = size(X)
        @assert length(ϑ) == d "Dimension mismatch: `X` has $d columns, but `ϑ` has length $(length(ϑ))."

        # Build the Noisy Covariance Matrix ---
        # We use the *fast* (mutating) version, as no AD is needed.
        K_noisy = build_covariance_matrix(X, ϑ, τ, σ2=σ2)
        
        # Define the GP Prior Distribution (a 0 mean MvNormal)
        prior_dist = MvNormal(zeros(N), K_noisy)
        
        # Draw One Sample
        y_sample = rand(prior_dist)
        
        return y_sample
    end



end