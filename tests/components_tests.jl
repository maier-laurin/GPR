#* testing the induviduall parts
1
using Plots
using Distributions
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
    N_obs = 50
    d_features = 1
    X_data = hcat(randn(N_obs)*5)

    # Define *true* hyperparameters
    ϑ_true = [1.0] # 1D ϑ
    τ_true = 1.0
    σ2_true = 0.1 

    # Generate a sample
    y_data = generate_gp_data(X_data, ϑ_true, τ_true, σ2_true)

    scatter(X_data, y_data)

    train_ml_model(X_data, y_data, num_restarts=50)
    
#---- testing ploting univariate GP 1D -----------------------------------------

    N_obs = 10
    d_features = 1
    X_data = hcat(randn(N_obs)*3)

    # Define *true* hyperparameters
    ϑ_true = [1.0] # 1D ϑ
    τ_true = 1
    σ2_true = 0.1 

    # Generate a sample
    y_data = generate_gp_data(X_data, ϑ_true, τ_true, σ2_true)
    #predict at some points using the real parameters
    x_vec = -8:0.01:8
    X = hcat(x_vec)
    y = predictive_distribution_marginal(X, X_data, y_data, ϑ_true, τ_true, σ2_true)

    #and visualice it
    begin
        p = plot_gp_heatmap(X[:,1], y, (-3, 3);y_resolution = 250, training_data = (X_data, y_data))
        plot!(p, size = (1020, 720))
        display(p)
    end

#---- testing ploting univariate GP 2D -----------------------------------------

    N_obs = 50
    d_features = 2
    X_data = hcat(randn(N_obs)*3, randn(N_obs)*3)

    # Define *true* hyperparameters
    ϑ_true = [1.0, 5.0] # 1D ϑ
    τ_true = 2
    σ2_true = 0.1 

    # Generate a sample
    y_data = generate_gp_data(X_data, ϑ_true, τ_true, σ2_true)
    #predict at some points using the real parameters
    x_vec = -5:0.05:5
    X = hcat(zeros(length(x_vec)), x_vec)
    y = predictive_distribution_marginal(X, X_data, y_data, ϑ_true, τ_true, σ2_true)

    #and visualice it
    begin
        p = plot_gp_heatmap(X[:,2], y, (-3, 3);y_resolution = 250, training_data = (X_data, y_data))
        plot!(p, size = (1020, 720))
        display(p)
    end

#---- mean field estimation ----------------------------------------------------

    N_obs = 100
    d_features = 2
    X_data = hcat(randn(N_obs)*3, randn(N_obs)*3)
    ϑ_true = [1.0, 5.0]
    τ_true = 2
    σ2_true = 0.1 
    y_data = generate_gp_data(X_data, ϑ_true, τ_true, σ2_true)
    
    a_hs = 0.45
    c_hs = 0.45

    μ, σ, result = train_mean_field(X_data, y_data, total_log_posterior_unnorm; a = a_hs, c = c_hs)

    labs = ["ϑ1", "ϑ2", "τ", "σ"]

    begin
        #plotting the posteriors all in one plot
        p = plot(title="mean field posteriors (exp(normal) = lognormal)", 
            xlabel="x", 
            ylabel="PDF", 
            xlims=(0, 6))
        for i in eachindex(μ)
            # LogNormal(μ, σ) takes the parameters of the underlying Normal
            d = LogNormal(μ[i], σ[i])
            plot(p, x -> pdf(d, x), 
                0, 6,  # Range to plot over (0 to 6)
                label=labs[i], 
                lw=2)
        end
        display(p)
    end
    #plot them induvidually
    begin
        i = 1
        d = LogNormal(μ[i], σ[i])
        p = plot(x -> pdf(d, x), 
                0, 3,
                label=labs[i],
                title = "length scale",
                lw=2)
        vline!(p, [1.0], linestyle=:dash, lw=2, label="Truth")
        display(p)
        #-------------
        i = 2
        d = LogNormal(μ[i], σ[i])
        p = plot(x -> pdf(d, x), 
                0, 20,
                label=labs[i],
                title = "length scale",
                lw=2)
        vline!(p, [5.0], linestyle=:dash, lw=2, label="Truth")
        display(p)
        #-------------
        i = 3
        d = LogNormal(μ[i], σ[i])
        p = plot(x -> pdf(d, x), 
                0, 4,
                label=labs[i],
                title = "Tau",
                lw=2)
        vline!(p, [2.0], linestyle=:dash, lw=2, label="Truth")
        display(p)
        #-------------
        i = 4
        d = LogNormal(μ[i], σ[i])
        p = plot(x -> pdf(d, x), 
                0, 0.5,
                label=labs[i],
                title = "Sigma",
                lw=2)
        vline!(p, [0.1], linestyle=:dash, lw=2, label="Truth")
        display(p)
    end
