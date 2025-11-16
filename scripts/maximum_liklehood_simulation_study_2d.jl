# as a first example we want to se how well ML recovers the true parameters, for that we
# simulate a sample from a Gaussian Process with known parameters, and than estimate the parameters from that sample


# over the unit circle, and then try to predict the choosen parameters with Maximum Liklehood, 
# and further Predict some out of training data sample points

#---- imports ------------------------------------------------------------------
    using GPR
    using Plots
    pgfplotsx()
    push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{newtxtext,newtxmath}")
    using PGFPlotsX
    using ProgressMeter
    using KernelDensity
    using LaTeXStrings
    using StatsBase
#---- building a function that does one such run -------------------------------
    function simulate_one_run(
        ϑ_true::AbstractVector{<:Real},
        τ_true::Real,
        σ2_true::Real,
        N::Int;
        num_restarts::Int = 3
    )
        # sampling a random patch
        d = length(ϑ_true)
        X = randn(N, d)
        y = generate_gp_data(X, ϑ_true, τ_true, σ2_true)

        # Run ML estimation on this synthetic dataset
        ml_result = train_ml_model(X, y; num_restarts=num_restarts, use_threads = false)

        # Extract estimated parameters
        ϑ_hat = ml_result.params_positive.ϑ
        τ_hat = ml_result.params_positive.τ
        σ2_hat = ml_result.params_positive.σ²

        return (
            ϑ_hat,
            τ_hat,
            σ2_hat
           )
    end
    
#---- visualicing the results --------------------------------------------------
    #deconstructing the vector
    function unpack_results(results::Vector{Tuple{Vector{Float64}, Float64, Float64}})
        N = length(results)
        d = length(results[1][1])  # length of ϑ-vector

        ϑ = zeros(Float64, N, d)
        τ = zeros(Float64, N)
        σ2 = zeros(Float64, N)

        for i in 1:N
            ϑ[i, :] .= results[i][1]
            τ[i]      = results[i][2]
            σ2[i]     = results[i][3]
        end

        return ϑ, τ, σ2
    end

    function visualice_results(results::Vector{Tuple{Vector{Float64}, Float64, Float64}}, ϑ_star::Vector{<:Real}, τ_star::Real, σ2_star::Real)
        ϑ_samp, τ_samp, σ2_samp = unpack_results(results)
        N, d = size(ϑ_samp)

        #* τ
        kd_τ = kde(τ_samp)
        begin
            pl_τ = plot(kd_τ.x, kd_τ.density,
                        xlabel = L"τ",
                        title  = "global variance τ",
                        legend = false,
                        xlims = (0,3)
                    )
            vline!(pl_τ, [τ_star], color = pl_τ.series_list[end].plotattributes[:seriescolor])
        end
        #* σ2
        kd_σ2 = kde(σ2_samp)
        begin
            pl_σ2 = plot(kd_σ2.x, kd_σ2.density,

                        xlabel = L"σ^2",
                        title  = "noise σ²",
                        legend = false,
                        xlims = (0,0.3)
                    )
            vline!(pl_σ2, [σ2_star], color = pl_σ2.series_list[end].plotattributes[:seriescolor])
        end
        #* ϑ
        kd_ϑ = Vector{UnivariateKDE}(undef, d)
        for i ∈ 1:d
            kd_ϑ[i] = kde(ϑ_samp[:, i])
        end
        begin
            pl_ϑ = plot(kd_ϑ[1].x, kd_ϑ[1].density,
                        xlabel = L"ϑ",
                        label  = latexstring("\\vartheta_1"),
                        title  = "length scales ϑ",
                        xlims = (0,5)
                    )
            vline!(pl_ϑ, [ϑ_star[1]], label = false, color = pl_ϑ.series_list[end].plotattributes[:seriescolor])
            for i ∈ 2:d
                plot!(pl_ϑ, kd_ϑ[i].x, kd_ϑ[i].density, label = latexstring("\\vartheta_$i"),)
                vline!(pl_ϑ, [ϑ_star[i]], label = false, color = pl_ϑ.series_list[end].plotattributes[:seriescolor])
            end
        end
        begin
            layout = @layout [a b; c{0.6h}]
            final_plot = plot(
                pl_τ, pl_σ2, pl_ϑ;
                layout = layout,
                ylabel = "",
                yticks = false,
                yaxis  = false,
                titlefontsize = FS_T,
                guidefontsize = FS_G,
                tickfontsize = FS_A,
                legendfontsize = FS_L,
                size = (cm_to_pt(TEXTWIDTH), cm_to_pt(9*TEXTWIDTH/16)),
            )
        end
        return final_plot
    end

#---- run this expiriment a couple of times for different number of samples ----

    #* run a parameter estimation on these fixed parameters
    ϑ_star = [1.0, 3.0, 0.5]
    τ_star = 1.0
    σ2_star = 0.1
    N_run = 1000 # the number of runs

    #* but for a few different number of samples
    begin
        N = 50 # the number of samples per run
        results = Vector{Tuple{Vector{Float64}, Float64, Float64}}(undef, N_run)
        pb = Progress(N_run, desc="Expiriments")
        Threads.@threads for i ∈ 1:N_run
            results[i] = simulate_one_run(ϑ_star, τ_star, σ2_star, N)
            next!(pb)
        end
        p1 = visualice_results(results, ϑ_star, τ_star, σ2_star)
    end
    begin
        N = 100 # the number of samples per run
        results = Vector{Tuple{Vector{Float64}, Float64, Float64}}(undef, N_run)
        pb = Progress(N_run, desc="Expiriments")
        Threads.@threads for i ∈ 1:N_run
            results[i] = simulate_one_run(ϑ_star, τ_star, σ2_star, N)
            next!(pb)
        end
        p2 = visualice_results(results, ϑ_star, τ_star, σ2_star)
    end
    begin
        N = 250 # the number of samples per run
        results = Vector{Tuple{Vector{Float64}, Float64, Float64}}(undef, N_run)
        pb = Progress(N_run, desc="Expiriments")
        Threads.@threads for i ∈ 1:N_run
            results[i] = simulate_one_run(ϑ_star, τ_star, σ2_star, N)
            next!(pb)
        end
        p3 = visualice_results(results, ϑ_star, τ_star, σ2_star)
    end
    #! you want to make shure you have something planed that will take a while and don't requre a computer (like a 2h lunch or so) before running this block...
    begin
        N = 1000 # the number of samples per run
        results = Vector{Tuple{Vector{Float64}, Float64, Float64}}(undef, N_run)
        pb = Progress(N_run, desc="Expiriments")
        Threads.@threads for i ∈ 1:N_run
            results[i] = simulate_one_run(ϑ_star, τ_star, σ2_star, N)
            next!(pb)
        end
        p4 = visualice_results(results, ϑ_star, τ_star, σ2_star)
    end

    lay = @layout [a b; c d]
    plot(p1, p2, p3, p4, layout = lay, size = (1980, 1080))

    #~savefig(p1, "plots/ml_parameter_recoverbility_50_samples.pdf")
    #~savefig(p2, "plots/ml_parameter_recoverbility_100_samples.pdf")
    #~savefig(p3, "plots/ml_parameter_recoverbility_250_samples.pdf")
    #~savefig(p4, "plots/ml_parameter_recoverbility_1000_samples.pdf")

