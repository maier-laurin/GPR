#* testing the induviduall parts

project_root = dirname(@__DIR__)

#---- kernel -------------------------------------------------------------------
    include(joinpath(project_root, "src/kernels.jl"))
    using .kernel
    
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
    @assert K_ad ≈ K_fast