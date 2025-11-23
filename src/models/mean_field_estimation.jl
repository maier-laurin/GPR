module MFModel
export train_mean_field

#---- Librarys and imports -----------------------------------------------------
    using LinearAlgebra
    using Statistics
    using Random
    using Optim
    using GPR

#---- function defenition ------------------------------------------------------
    """
        train_mean_field(
            X::Matrix{<:Real},
            y::AbstractVector{<:Real},
            log_post_func::Function;
            a::Real,
            c::Real,
            n_samples::Int=20,
            max_iters::Int=1000
        )

    Performs Mean Field Variational Inference (ADVI) to approximate the posterior
    of the GP kernel parameters.
    For that it approximates the posterior p(p_log | y) with a diagonal Gaussian
    q(p_log) = N(μ_vi, diag(σ_vi²))

    # Arguments
    - `X`: Input matrix (N x d).
    - `y`: Observation vector (N).
    - `log_post_func`: The user-provided function `total_log_posterior_unnorm`.

    # Keyword Arguments
    - `a`, `c`: Hyperparameters passed to the posterior function.
    - `n_samples`: Number of Monte Carlo samples used to estimate the ELBO.
    (Higher = more accurate, slower).
    - `max_iters`: Maximum iterations for the L-BFGS optimizer.

    # Returns
    Tuple with:
    - `μ`: The posterior mean vector of the parameters in log-space.
    - `σ`: The posterior standard deviation vector of the parameters in log-space.
    - `result`: The full Optim.jl result object.
    """
    function train_mean_field(
        X::Matrix{<:Real},
        y::AbstractVector{<:Real},
        log_post_func::Function;
        a::Real,
        c::Real,
        n_samples::Int=20,
        max_iters::Int=1000
    )
        N, d = size(X)
        n_params = d + 2 #the length scales plus τ and σ

        #* We need to optimize 2 * n_params:
        # First half: Variational Means (mu)
        # Second half: Variational Log-Standard Deviations (log_sigma)
        
        #* initialization according to the following educadet guesses
        # Initialize means close to 0.0 (unit length scales/variance in real space)
        # Initialize log_stds to -1.0 (small initial uncertainty)
        initial_params = vcat(zeros(n_params), fill(-1.0, n_params))

        #To speed up ELBO we freeze the random noise ε i.e. sampling it beforhand. This makes the ELBO function deterministic with respect to the 
        # variational parameters, allowing us to use  the faster Quasi-Newton methods (L-BFGS) instead of slow SGD.
        base_samples = randn(n_params, n_samples)

        #*Define the Negative ELBO
        function neg_elbo(variational_flat)
            # Unpack parameters
            μ_vi = variational_flat[1:n_params]
            log_σ_vi = variational_flat[n_params+1:end]
            σ_vi = exp.(log_σ_vi)
            # Using Reparameterization Trick: ξ = μ + σ * ε
            avg_log_joint = 0.0
            for i in 1:n_samples
                # Get pre-sampled noise
                ε = view(base_samples, :, i)
                # Transform to parameter space
                p_log_sample = μ_vi .+ σ_vi .* ε

                # evaluate the pased log posterior function
                val = log_post_func(p_log_sample, X, y; a=a, c=c)

                if val == -Inf
                    #§ When defining the log posterior i thought returning -∞ for invalid resons is very elegant
                    #§ long story short it breaks everything, so i approximate \infty with 10^10
                    return 1e10 
                end
                avg_log_joint += val
            end
            avg_log_joint /= n_samples

            # We maximize Entropy H, so in Neg-ELBO we subtract it.
            entropy = sum(log_σ_vi) + 0.5 * n_params * (log(2π) + 1)

            # ELBO = E[log p] + H(q)
            return -(avg_log_joint + entropy)
        end

        # Optimiz using L-BFGS
        # We use autodiff=:forward to automatically calculate gradients of neg_elbo
        opt_res = optimize(
            neg_elbo, 
            initial_params, 
            LBFGS(), 
            Optim.Options(
                iterations=max_iters, 
                show_trace=true,
                show_every=50
            );
            autodiff = :forward
        )

        # returnstruckture 
        final_vars = Optim.minimizer(opt_res)
        mu_vi_final = final_vars[1:n_params]
        sigma_vi_final = exp.(final_vars[n_params+1:end])

        return (μ = mu_vi_final, σ = sigma_vi_final, result = opt_res)
    end
end