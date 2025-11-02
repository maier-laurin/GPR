#* Maximum Liklehood Model

#---- Librarys and imports -----------------------------------------------------
    using Optim
    include("kernel_matrix.jl")
    include("prior_distribution.jl")
    include("GPR_log_liklihood.jl")

#---- function defenition ------------------------------------------------------
    """
        train_ml_model(
            X::Matrix{<:Real}, 
            y::AbstractVector{<:Real}; 
            num_restarts::Int = 10
        ) -> NamedTuple

    Trains a GPR model using Maximum Likelihood (ML).

    This function finds the parameters {ϑ, τ, σ²} that maximize the 
    `gpr_log_likelihood`. It uses the L-BFGS optimizer and performs
    multiple restarts to find a good optimum.

    # Arguments
    - `X`: The N x d input data matrix.
    - `y`: The N-dimensional observation vector.

    # Keyword Arguments
    - `num_restarts`: The number of times to restart the optimization
                    from a new random starting point.

    # Returns
    - A `NamedTuple` containing:
        - `params_log`: The optimized parameters in *log-space*.
        - `params_positive`: The optimized parameters in *positive-space* (ϑ, τ, σ²).
        - `min_nll`: The minimum Negative Log-Likelihood (NLL) value found.
    """
    function train_ml_model(
        X::Matrix{<:Real}, 
        y::AbstractVector{<:Real}; 
        num_restarts::Int = 10
    )
        N, d = size(X)
        
        # This is the objective function for Optim.jl to *minimize*.
        # It takes a single vector `p` of *unconstrained* parameters.
        function nll_objective(p_log::AbstractVector{<:Real})::Real
            try
                # --- 1. Reparameterization (Unconstrained -> Constrained) ---
                # We use `exp` to ensure all parameters are positive.
                ϑ = exp.(p_log[1:d])
                τ = exp(p_log[d+1])
                σ² = exp(p_log[d+2])

                # --- 2. Build Covariance Matrix (Fast Version) ---
                # We don't need AD here, so we use the fast, mutating version.
                K_noisy = build_covariance_matrix(X, ϑ, τ, σ2=σ²)
                
                # --- 3. Calculate NLL ---
                # We return the *negative* log-likelihood for minimization.
                return -gpr_log_likelihood(y, K_noisy)
                
            catch e
                # If `cholesky` fails (e.g., matrix not PD), return Inf
                # to tell the optimizer this is a bad region.
                if isa(e, PosDefException)
                    return Inf
                else
                    rethrow(e)
                end
            end
        end
        
        # --- Optimization Loop ---
        best_nll = Inf
        best_params_log = Vector{Float64}(undef, d + 2)
        
        println("Starting ML training with $num_restarts restarts...")
        
        for i in 1:num_restarts
            # 1. Random start for *log-parameters*
            #    `randn` centers starting points around 1.0 in positive-space
            p_start_log = randn(d + 2)
            
            # 2. Run the optimizer
            result = optimize(
                nll_objective, 
                p_start_log, 
                LBFGS(),
                Optim.Options(g_tol = 1e-5, iterations = 1000)
            )
            
            # 3. Store if it's the best result
            if Optim.minimum(result) < best_nll
                best_nll = Optim.minimum(result)
                best_params_log = Optim.minimizer(result)
            end
            print(".")
        end
        println("\nML training complete. Best NLL: $best_nll")
        
        # Return the best results
        return (
            params_log = best_params_log,
            params_positive = (
                ϑ = exp.(best_params_log[1:d]),
                τ = exp(best_params_log[d+1]),
                σ² = exp(best_params_log[d+2])
            ),
            min_nll = best_nll
        )
    end

#---- Testing ------------------------------------------------------------------
#=
    # 1. Define dummy data
    N_obs = 100
    N_features = 3
    X_data = randn(N_obs, N_features)
    y_data = sin.(X_data[:, 1]*2) .* 0.5 + randn(N_obs) .* 0.1
    scatter(X_data[:,1], y_data)
    
    ml_result = train_ml_model(X_data, y_data, num_restarts=10)

    println("\n--- Optimized ML Parameters ---")
    println("ϑ (positive): ", ml_result.params_positive.ϑ)
    println("τ (positive): ", ml_result.params_positive.τ)
    println("σ² (positive): ", ml_result.params_positive.σ²)
=#