module GPR

#* components
include("components/kernels.jl")
using .Kernels
export build_covariance_matrix, build_covariance_matrix_ad, se_kernel

include("components/likelihood.jl")
using .Likelihood
export gpr_log_likelihood, gpr_log_likelihood_unnorm

include("components/special_special_functions.jl")
using .SpecialSpecialFunctions
export kummer_U

include("components/priors.jl")
using .Priors
export log_prior_unnorm

include("components/posterior.jl")
using .Posteriors
export total_log_posterior_unnorm

include("components/data_generation.jl")
using .DataGeneration
export generate_gp_data

include("components/predictions.jl")
using .Predict
export predictive_distribution, predictive_distribution_marginal

include("components/visuals.jl")
using .Visuals
export extract_marginal_distributions, get_marginal_distributions, plot_gp_heatmap


#* Include the model files

include("models/maximum_likelihood.jl")
using .MLModel
export train_ml_model

include("models/mean_field.jl")
using .MFModel
export train_mf_model

end # module