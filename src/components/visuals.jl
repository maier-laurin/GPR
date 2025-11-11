#* Unnormaliced log posterior
module Visuals
    
#---- Librarys and imports -----------------------------------------------------
    using Plots
    using Distributions
    using KernelDensity
    using LinearAlgebra
    using LaTeXStrings
    using ..Predict
    export extract_marginal_distributions, get_marginal_distributions, plot_gp_heatmap
#---- function defenition ------------------------------------------------------
    #* helper function for GP with fixed parameter
    """
        extract_marginal_distributions(
            predictive_dist::MvNormal
        ) -> Vector{Normal}

    Takes a Multivariate Normal posterior predictive distribution and
    returns a Vector of its 1D marginal distributions.

    i.e. if we want to plot a gaussian process with fixed (skalar) parameters for the Covariance function,
    we can give it the big MV normal of the Posterior (thats wy fixed parameters) and the function returns the univariate Normals
    for each point (dimension).
    """
    function extract_marginal_distributions(
        predictive_dist::MvNormal
    )
        
        # Get the mean vector (μ)
        μ = mean(predictive_dist)
        
        # Get the marginal variances (the diagonal of Σ)
        #§ in theory the variance could be 0 which would be bad for the plotting, but on the other hand than the Matrix wouldnt be positive definite, and colesky would have failed already
        σ2_diag = diag(cov(predictive_dist))
        
        # 3. Get the marginal standard deviations
        σ_diag = sqrt.(σ2_diag)
        
        # Create the Vector of 1D Normal distributions
        return [Normal(μ[i], σ_diag[i]) for i ∈ eachindex(μ)]
    end

    #* helper function for GP with distributed parameters aka sampled posterior distribution
    """
        get_marginal_distributions(
            X_test::Matrix{<:Real},
            X_train::Matrix{<:Real},
            y_train::AbstractVector{<:Real},
            q_xi::Distribution, # Our trained posterior: MvNormal or Flow
            M_samples::Int = 1000
        ) -> Vector{UnivariateKDE}

    In cases where the parameters of the Covariance function arn't fixed, but given as a distribution,
    i.e. they are the result of a baysian analysis, we can't give the posterior of theprocess as a closed from
    MV Gaussian. Therefor we need to estimate it via Monte carlo. We then can perform a Kernel densety estimation
    of the underlying distribution, and build our vector of distibutions for plotting by doing that for each
    x in X_test. X_train and y_train are needed for predicting, and q_xi is the infered distribution for the parameters of the 
    covariance function, so in our cases a MvNormal, or a Flow (just needs to implement the Distribution.jl Interface).
    M_sample is the number of points sampled per x ∈ X_train.

    Returns a vector of `UnivariateKDE` objects, one for each test point.
    """
    function get_marginal_distributions(
        X_test::Matrix{<:Real},
        X_train::Matrix{<:Real},
        y_train::AbstractVector{<:Real},
        q_xi::Distribution,
        M_samples::Int = 1000
    )::Vector{UnivariateKDE}

        N_test, d_test = size(X_test)
        N_train, d_train = size(X_train)
        d = d_train # The number of ϑ parameters

        @assert d_test == d_train "Feature dimension mismatch."
        
        # Create a "sample bin" for each test point
        #   each row `i` will hold all M samples for `X_test[i]`.
        y_samples_all = Matrix{Float64}(undef, N_test, M_samples)

        #* effectifly we don't sample M_samples * N_test "independent" points, but just
        #* N_test "independent" points from M_sample different realisations of the GP
        # Run the Monte Carlo Loop
        #§ trivially parallelicable 
        # the content of this loop is independend for each collumn of the Matrix

        Threads.@threads for m ∈ 1:M_samples
            
            # Draw one sample of unconstrained log-parameters
            p_log_sample = rand(q_xi)
            
            # Transform to non log space
            ϑ_sample = exp.(p_log_sample[1:d])
            τ_sample = exp(p_log_sample[d+1])
            σ2_sample = exp(p_log_sample[d+2])

            # Get the predictive distribution *conditional* on this one sample
            pred_dist_m = predictive_distribution(
                X_test, X_train, y_train, 
                ϑ_sample, τ_sample, σ2_sample
            )
            
            # Draw one y_test vector from it
            y_samples_all[:, m] = rand(pred_dist_m)
        end
        
        # Create KDEs for each test point
        #§ independent for each row
        kde_distributions = Vector{UnivariateKDE}(undef, N_test)
        Threads.@threads for i in 1:N_test
            # Get all samples for the i-th test point
            y_i_samples = view(y_samples_all, i, :)
            
            # Create the Kernel Density Estimate
            kde_distributions[i] = kde(y_i_samples) # a kde is a subtype of a Distribution
        end
        
        return kde_distributions
    end


    #* plot a 1D GP
    """
        plot_gp_heatmap(
            X_values::AbstractVector{<:Real},
            Y_distributions::AbstractVector{<:Distribution},
            y_range::Tuple{Float64, Float64};
            y_resolution::Int = 100,
            training_data::Union{Nothing, Tuple} = nothing,
            plot_title::String = "GP Posterior Predictive Density"
        )

    Generates a 2D heatmap of the 1D posterior predictive distributions.

    # Arguments
    - `X_values`: The vector of x-axis locations (N_test points).
    - `Y_distributions`: The vector of 1D distributions (N_test objects),
                        e.g., `Vector{Normal}` or `Vector{UnivariateKDE}`.
    - `y_range`: A tuple `(min, max)` defining the y-axis range to plot.

    # Keyword Arguments
    - `y_resolution`: The number of vertical "slices" for the heatmap.
    - `training_data`: An optional tuple `(X_train, y_train)` to scatter
                    on top of the plot.
    - `plot_title`: The title for the plot.
    """
    function plot_gp_heatmap(
        X_values::AbstractVector{<:Real},
        Y_distributions::AbstractVector{<:Distribution},
        y_range::Tuple{Real, Real};
        y_resolution::Int = 100,
        training_data::Union{Nothing, Tuple} = nothing,
        plot_title::String = "GP Posterior Predictive Density"
    )
        
        # Define the plot grids
        y_grid = range(y_range[1], y_range[2], length=y_resolution)
        x_grid = X_values
        
        n_x = length(x_grid)
        n_y = length(y_grid)
        
        # Create the heatmap data matrix
        heatmap_data = Matrix{Float64}(undef, n_y, n_x)
        
        # Loop over the matrix and fill with PDF values
        Threads.@threads for i ∈ eachindex(x_grid)
            # Get the corresponding distribution
            dist = Y_distributions[i]
            
            #calculate the nProbability values at each gridpoint
            for (j, y) in enumerate(y_grid)
                heatmap_data[j, i] = pdf(dist, y)
            end
        end
        
        # Create the base heatmap plot
        p = heatmap(
            x_grid,
            y_grid,
            heatmap_data,
            xlabel = L"x",
            ylabel = L"f(x)",
            title = plot_title,
            color = cgrad([:white, :blue, :darkblue]), #TODO find domething aestheticly pleasing
            colorbar_title = "\nProbability Density",
            legend = :topleft
        )
        
        # Plot the mean prediction
        mean_line = mean.(Y_distributions) #the distribution Interface promises a means methode for all distribution objects
        plot!(
            p,
            x_grid,
            mean_line,
            label = "Mean Prediction",
            color = :darkblue,
            linewidth = 1.5,
            linestyle = :dot
        )
        
        # Plot training data on top
        #TODO add assertions to check for correct types (Vectors of same length)
        if training_data !== nothing
            X_train, y_train = training_data
            scatter!(
                p,
                X_train,
                y_train,
                label = "Samples",
                color = :darkorange,
                markersize = 5,
                markerstrokecolor = :white
            )
        end

        return p
    end

end