#* mean field field model
module MFModel
export train_mf_model

#---- Librarys and imports -----------------------------------------------------
    using Turing
    using Distributions
    using DistributionsAD
    using LinearAlgebra
    using Zygote
    using ADTypes
    using AdvancedVI
    using ..Kernels
    using ..Priors
    using ..Likelihood
    using ..Posteriors


#---- function defenition ------------------------------------------------------
    """
        gpr_model_vi(X, y, a, c, d)

    Turing.jl `@model` for Mean-Field Variational Inference.

    This model defines a (d+2)-dimensional unconstrained parameter
    vector `p_log` (representing [log(ϑ)..., log(τ), log(σ²)]).

    It uses the `Turing.@addlogprob!` macro to set the target density
    to our `total_log_posterior_unnorm` function.
    """
    @model function gpr_model_vi(
        X::Matrix{<:Real},
        y::AbstractVector{<:Real},
        a::Real,
        c::Real,
        d::Int
    )
        T = promote_type(eltype(X), eltype(y), typeof(a), typeof(c), typeof(d))
        # Defining a simple base prior for the unconstrained parameters,
        # The VI will find an approximation *to this parameter vector*.
        base_prior = MvNormal(zeros(d + 2), T(10.0) * I)
        # the ~ registers the parameters that define the base distribution as the ons to optimice
        # so in essence we want to optimice over a μ and a Σ, and as an initiall guess (x_0) for the optimicer
        # we have 0 & 10I (just conceptually, because its stochastic sampling optimication, but for the understanding its o.k. to think so)
        p_log ~ base_prior
        
        # with this so to say trick we swap the target distribution
        # it's what tells the optimizer: "When you're deciding how to update μ and Σ, don't compare f* to the base_prior. 
        #   Compare it to the real target distribution
        Turing.@addlogprob!(
            total_log_posterior_unnorm(p_log, X, y, a=a, c=c) - 
            logpdf(base_prior, p_log)
        )
    end

    function train_mf_model(
        X::Matrix{<:Real},
        y::AbstractVector{<:Real};
        a::Real,
        c::Real,
        vi_samples::Int = 1000
    )
        N, d = size(X)
        
        # Instantiate our Turing model (the "target")
        model = gpr_model_vi(X, y, a, c, d)
        
        # create the mean-field Gaussian approximation
        q = q_meanfield_gaussian(model, )
        
        # Define the algorithm to fit "q" to "model"
        alg = ADVI(ADTypes.AutoZygote())

        println("Starting Mean-Field (a=$a, c=$c) training...")
        
        # Run VI
        approximate_posterior = vi(
            model, 
            q,
            alg,
            vi_samples
        )
        
        println("Mean-Field training complete.")
        
        return approximate_posterior
    end

end