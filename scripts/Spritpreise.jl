# testing the maximum liklehood methode on a real world data set, i.e. the Prices of petrols at the gas stations in Austria
# data scraped from econtrol on 11/12/2025 the goal was to determin the parameters of a GPR Model with exponential kernel, 
# and predictit the mean price at random/ grided locations all over austria...
#! experiment failed

using GPR
using Plots
using DataFrames
using CSV
using Statistics

#---- data loading and preping -------------------------------------------------
    sp = CSV.read("data/Spritpreise.csv", DataFrame)
    scatter(sp[!,:y], sp[!,:x], marker_z = sp[!,:p],
            axis = nothing,
            border = :none,
            legend = false,
            title = "gasstations in Austria",
            size = (1920, 1020),
            color = cgrad(["#83def2", "#729ff2", "#4f88f0", "#4f88f0", :darkblue]),
            markersize = 8,
            markerstrokewidth = 0)
    #* coordinates are in the Lambert Projection, recentered to Radstadt as the origin
    # rescaling to kilometers
    X = Matrix(sp[!,[:x, :y]])./1000
    # rescaling to cents, and demeaning
    y = Vector(sp[!, :p])*100
    y_mean = mean(y)
    y .-= y_mean
#---- maximum liklehood estimation of parameters -------------------------------
    #! takes vary very long and does not leed to promesing results
    ml_result = train_ml_model(X, y, num_restarts=22)
    ϑ = ml_result[:params_positive][:ϑ]
    τ = ml_result[:params_positive][:τ]
    σ2 = ml_result[:params_positive][:σ²]
    #* results of one run:
    #~ ϑ = [0.00015346522032807285, 0.0002716948807641528]
    #~ τ = τ = 4.116633519591036e-108
    #~ σ² = 231.7430264323427
#---- predicting a grid --------------------------------------------------------
    x_range = -100:5:175
    y_range = -300:5:270
    grid = reduce(hcat, [[x, y] for x in x_range for y in y_range])'
    predictions = predictive_distribution_marginal(grid, X, y, ϑ, τ, σ2)
    pred_mean = mean.(predictions)
#---- plotting the results -----------------------------------------------------
    x_ticks = unique(sort(grid[:, 1]))
    y_ticks = unique(sort(grid[:, 2]))
    n_x = length(x_ticks)
    n_y = length(y_ticks)

    x_index_map = Dict(val => i for (i, val) in enumerate(x_ticks))
    y_index_map = Dict(val => i for (i, val) in enumerate(y_ticks))

    Z = fill(NaN, (n_y, n_x))
    for i ∈ eachindex(pred_mean)
        x_val = grid[i, 1]
        y_val = grid[i, 2]
        z_val = pred_mean[i]
        row_idx = y_index_map[y_val]
        col_idx = x_index_map[x_val]
        Z[row_idx, col_idx] = z_val
    end
    
    heatmap(x_range, y_range, Z,
            axis = nothing,
            border = :none,
            legend = false,
            title = "petrolprices in Austria",
            size = (1920, 1020),
            color = cgrad(["#83def2", "#729ff2", "#4f88f0", "#4f88f0", :darkblue]))