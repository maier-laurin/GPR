#* mean field field model

#---- Librarys and imports -----------------------------------------------------
    using Turing
    using Distributions
    using DistributionsAD
    include("kernel_matrix.jl")
    include("prior_distribution.jl")
    include("GPR_log_liklihood.jl")