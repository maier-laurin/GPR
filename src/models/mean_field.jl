#* mean field field model
module MFModel
export train_mf_model

#---- Librarys and imports -----------------------------------------------------
    using Turing
    using Distributions
    using DistributionsAD
    using ..Kernels
    using ..Priors
    using ..Likelihood



#---- function defenition ------------------------------------------------------
    """
        gpr_model_vi(X, y, a, c, d)

    Turing.jl `@model` for Mean-Field Variational Inference.

    This model defines a (d+2)-dimensional unconstrained parameter
    vector `p_log` (representing [log(ϑ)..., log(τ), log(σ²)]).

    It uses the `Turing.@addlogprob!` macro to set the target density
    to our custom `total_log_posterior_unnorm` function, effectively
    "swapping" the simple base prior for our complex target.
    """
    @model function gpr_model_vi(
        X::Matrix{<:Real},
        y::AbstractVector{<:Real},
        a::Real,
        c::Real,
        d::Int
    )
        # 1. Define a simple base prior for our unconstrained parameters.
        #    The VI will find an approximation *to this parameter vector*.
        base_prior = MvNormal(zeros(d + 2), 10.0 * I)
        p_log ~ base_prior
        
        # 2. "Swap" the base prior for our custom log-posterior.
        #    We subtract the logpdf of the base_prior (which `~` added)
        #    and add the logpdf of our *actual* target distribution.
        Turing.@addlogprob!(
            total_log_posterior_unnorm(p_log, X, y, a=a, c=c) - 
            logpdf(base_prior, p_log)
        )
    end

    """
        train_mf_model(
            X::Matrix{<:Real},
            y::AbstractVector{<:Real};
            a::Real,
            c::Real,
            vi_samples::Int = 1000
        ) -> MultivariateDistribution

    Trains a GPR model using Mean-Field Variational Inference (ADVI).

    This function implements the "MF: HS" and "MF: TG" models from
    the paper (depending on the `a` and `c` values passed).

    # Arguments
    - `X`: The N x d input data matrix.
    - `y`: The N-dimensional observation vector.

    # Keyword Arguments
    - `a`: The first hyperparameter (e.g., 0.5 for HS, 0.1 for TG).
    - `c`: The second hyperparameter (e.g., 0.5 for HS, 0.1 for TG).
    - `vi_samples`: The number of samples to draw for the VI optimization.

    # Returns
    - A `MultivariateDistribution` (specifically, a `MvNormal`) which
    represents the mean-field approximation of the posterior 
    distribution *over the log-parameters `p_log`*.
    """
    function train_mf_model(
        X::Matrix{<:Real},
        y::AbstractVector{<:Real};
        a::Real,
        c::Real,
        vi_samples::Int = 1000
    )
        N, d = size(X)
        
        # 1. Instantiate our Turing model
        model = gpr_model_vi(X, y, a, c, d)
        
        # 2. Run the Mean-Field VI (ADVI = Automatic Differentiation VI)
        #    ADMeanField specifies a factorized (diagonal) Gaussian
        #    approximation, which is exactly what "mean-field" means.
        println("Starting Mean-Field (a=$a, c=$c) training...")
        approximate_posterior = vi(
            model, 
            ADVI{ADMeanField}(),
            vi_samples # Number of gradient samples
        )
        println("Mean-Field training complete.")
        
        # 3. Return the resulting distribution
        #    This object contains the means and std-devs of the
        #    approximating Gaussian distribution.
        return approximate_posterior
    end

end