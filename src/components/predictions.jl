#*

module Predict
export predictive_distribution

#---- Librarys and imports -----------------------------------------------------
    
#---- functions ----------------------------------------------------------------
    #* prdictive distribution for fixed parameter

    function predictive_distribution(
                X_test, 
                X_train, 
                y_train, 
                ϑ, 
                τ, 
                σ2
            )
        return MvNormal(0, 0)
    end

end