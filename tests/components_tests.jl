#* testing the induviduall parts

using Plots
using GPR

#---- kernel -------------------------------------------------------------------
    
    # 1. Define dummy data
    N_obs = 10000
    N_features = 5
    X_data = randn(N_obs, N_features)
    theta_params = [0.5, 1.0, 0.0, 5.0, 0.02]
    tau_param = 1.5
    sigma2_noise = 0.1

    @time K_ad = build_covariance_matrix_ad(X_data, theta_params, tau_param, σ2=sigma2_noise)
    @time K_fast = build_covariance_matrix(X_data, theta_params, tau_param, σ2=sigma2_noise)

    # They should be identical
    K_ad ≈ K_fast

#---- priors ------------------------------------------------------------------

    # Define hyperparameters
    a_val = 0.1
    c_val = 0.1 
    λ_val = 10.0 

    # Define dummy parameters
    ϑ_params = [0.2, 1.5, 0.01]
    τ_param  = 0.5
    σ2_param = 0.1


    # Test the total joint log-prior
    total_log_prior = log_prior_unnorm(
        ϑ_params, τ_param, σ2_param, 
        a=a_val, c=c_val, λ=λ_val
    )
    plot(x -> log_prior_unnorm(
        ϑ_params, x, σ2_param, 
        a=a_val, c=c_val, λ=λ_val
    ), 0.01, 2)

#---- maximum_likelihood model -------------------------------------------------

    # Define dummy data
    N_obs = 100
    N_features = 3
    X_data = randn(N_obs, N_features)
    y_data = sin.(X_data[:, 1]*2) .* 0.5 + randn(N_obs) .* 0.1
    scatter(X_data[:,1], y_data)
    
    ml_result = train_ml_model(X_data, y_data, num_restarts=25)

    println("\n--- Optimized ML Parameters ---")
    println("ϑ (positive): ", ml_result.params_positive.ϑ)
    println("τ (positive): ", ml_result.params_positive.τ)
    println("σ² (positive): ", ml_result.params_positive.σ²)

#---- posterior ------------------------------------------------------------------

        p_log_optimized = ml_result.params_log

    # 2. Define the prior settings
        a_hs = 0.500001 #? why is does it evaluate to Inf for a = 0.5?
        c_hs = 0.5

        a_tg = 0.1
        c_tg = 0.1

    # 3. Calculate the total log-posterior
        log_post_hs = total_log_posterior_unnorm(
            p_log_optimized, X_data, y_data, a=a_hs, c=c_hs
        )

        log_post_tg = total_log_posterior_unnorm(
            p_log_optimized, X_data, y_data, a=a_tg, c=c_tg
        )

    println("Log-Posterior (Horseshoe): ", log_post_hs)
    println("Log-Posterior (Triple Gamma): ", log_post_tg)
    
    y = map(x -> exp(total_log_posterior_unnorm(p_log_optimized, [0 0 0.0], [x], a=a_tg, c=c_tg)), -5:0.1:5)
    plot(-5:0.1:5, y)

#---- data generation ----------------------------------------------------------

    # Define the "world"
    N_obs = 200
    d_features = 1
    X_data = hcat(randn(N_obs)*5)

    # Define *true* hyperparameters
    ϑ_true = [1.0] # 1D ϑ
    τ_true = 1.0
    σ²_true = 0.1 

    # Generate a sample
    y_data = generate_gp_data(X_data, ϑ_true, τ_true, σ²_true)

    scatter(X_data, y_data)

    train_ml_model(X_data, y_data, num_restarts=50)
    

#---- mean field estimation ----------------------------------------------------

    a_hs = 0.5
    c_hs = 0.5
    mf_posterior_hs = train_mf_model(X_data, y_data, a=a_hs, c=c_hs)

    println("\n--- MF: HS (Horseshoe) Result ---")
    println("Approximation type: ", typeof(mf_posterior_hs))
    println("Posterior Mean (log-space): ", round.(mean(mf_posterior_hs), digits=3))
    println("Posterior StdDev (log-space): ", round.(sqrt.(diag(cov(mf_posterior_hs))), digits=3))

    # --- 2. Train Model 3: Mean-Field Triple Gamma (MF: TG) ---
    #    Corresponds to a=0.1, c=0.1 [cite: 410]
    a_tg = 0.1
    c_tg = 0.1
    mf_posterior_tg = train_mf_model(X_data, y_data, a=a_tg, c=c_tg)

    println("\n--- MF: TG (Triple Gamma) Result ---")
    println("Approximation type: ", typeof(mf_posterior_tg))
    println("Posterior Mean (log-space): ", round.(mean(mf_posterior_tg), digits=3))
    println("Posterior StdDev (log-space): ", round.(sqrt.(diag(cov(mf_posterior_tg))), digits=3))

    # --- 3. How to USE this result ---
    #    To make a prediction, you draw samples from this distribution
    #    and pass them to our prediction function.

    println("\n--- Example: Drawing one sample from MF: HS posterior ---")
    p_log_sample = rand(mf_posterior_hs)
    N_test, d_test = size(X_data) # Just for demonstration

    # Transform back to positive-space
    ϑ_sample = exp.(p_log_sample[1:d_test])
    τ_sample = exp(p_log_sample[d_test+1])
    σ²_sample = exp(p_log_sample[d_test+2])

    println("Sample ϑ: ", round.(ϑ_sample, digits=3))
    println("Sample τ: ", round(τ_sample, digits=3))
    println("Sample σ²: ", round(σ²_sample, digits=3))

    # (You would then pass these to `predict_posterior_distribution`)