#* Maximum Liklehood Model

module MLModel
export train_ml_model
#---- Librarys and imports -----------------------------------------------------
    using Optim
    using LinearAlgebra
    using ..Kernels
    using ..Priors
    using ..Likelihood   

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
        num_restarts::Int = 10,
        use_threads::Bool = true
    )
        T = promote_type(eltype(X), eltype(y))
        N, d = size(X)
        
        # This is the objective function for Optim.jl to *minimize*.
        # It takes a single vector `p` of *unconstrained* parameters.
        function nll_objective(p_log::AbstractVector{<:Real})
            try
                # Reparameterization (Unconstrained -> Constrained)
                # We use `exp` to ensure all parameters are positive.
                #§ where singularity could lead to problems we artificially set the valid domain to (ϵ, ∞)
                ϵ = 1e-6
                ϑ = exp.(p_log[1:d])
                τ = exp(p_log[d+1]) +  ϵ
                σ² = exp(p_log[d+2]) +  ϵ

                # Build Covariance Matrix (Fast Version)
                # We don't need AD here, so we use the fast, mutating version.
                K_noisy = build_covariance_matrix(X, ϑ, τ, σ2=σ²)
                
                # Calculate NLL
                # We return the *negative* log-likelihood for minimization.
                return -gpr_log_likelihood(y, K_noisy)
                
            catch e
                # If `cholesky` fails (e.g., matrix not PD), return Inf
                # to tell the optimizer this is a bad region.
                if isa(e, PosDefException)
                    T = promote_type(eltype(X), eltype(y))
                    return T(Inf)
                else
                    rethrow(e)
                end
            end
        end
        
        # --- Optimization Loop ---
        best_nll = Inf
        best_params_log = Vector{T}(undef, d + 2)
        nthreads = Threads.nthreads()
        if use_threads
            println("Starting ML training with $num_restarts restarts on $nthreads awailable threads...")
        end
        # the restart loop is trivially paralicable, we only need to be carefull about race contitions
        # therefor we save all solutions and find the best later
        all_results = Vector{Any}(undef, num_restarts)
        # flaged multithreading
        if use_threads
            Threads.@threads for i ∈ 1:num_restarts
                # Random start for *log-parameters* log around 0 -> exp around 1
                p_start_log = randn(T, d + 2)
                
                # Run the optimizer
                result_i = optimize(
                    nll_objective, 
                    p_start_log, 
                    LBFGS(),
                    Optim.Options(g_tol = 1e-5, iterations = 10000)
                )

                all_results[i] = result_i
                print(".")
            end
        else
            # in the unthreaded case we assume a outher process is threaded so the status updates here dont make much sense.
            for i ∈ 1:num_restarts
                p_start_log = randn(T, d + 2)
                result_i = optimize(
                    nll_objective, 
                    p_start_log, 
                    LBFGS(),
                    Optim.Options(g_tol = 1e-5, iterations = 10000)
                )
                all_results[i] = result_i
            end
        end

        obj_values = Vector{T}(undef, num_restarts)
        for i ∈ eachindex(all_results)
            obj_values[i] = Optim.minimum(all_results[i])
            if Optim.minimum(all_results[i]) < best_nll
                best_nll = Optim.minimum(all_results[i])
                best_params_log = Optim.minimizer(all_results[i])
            end
        end
        sort!(obj_values)
        if use_threads
            println("\nML training complete.")
        end
        # Return the best results
        ϵ = 1e-6
        return (
            params_log = best_params_log,
            params_positive = (
                ϑ = exp.(best_params_log[1:d]),
                τ = exp(best_params_log[d+1]) + ϵ,
                σ² = exp(best_params_log[d+2]) + ϵ
            ),
            objective_dist = obj_values,
            min_nll = best_nll
        )
    end

end