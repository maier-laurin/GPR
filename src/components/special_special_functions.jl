#* kummer_U

module SpecialSpecialFunctions
export kummer_U

#---- Librarys and imports -----------------------------------------------------
    using SpecialFunctions # Gamma function for `kummer_U`
    using HypergeometricFunctions # Parts for `kummer_U`

#---- additionall implementation -----------------------------------------------

    #TODO check for correctness or build something different 
    #! this is just the result of a 2 hour discussion with a large language model about 
    #! if theres a implementation of the confluent hypergeometric function of the second
    #! kind in julia that does not call C++ over GLS because that screams trouble with AD. 
    #* And this is the best we came up with
    """
        kummer_U(a, b, z)

    Calculates the confluent hypergeometric function of the second kind, U(a, b, z),
    using its definition in terms of the function of the first kind, ₁F₁ (Kummer's M).
    """
    function kummer_U(a::Number, b::Number, z::Number)
        # The identity requires two terms
        
        # Term 1: (Γ(1-b) / Γ(a-b+1)) * ₁F₁(a; b; z)
        term1 = (gamma(1 - b) / gamma(a - b + 1)) * _₁F₁(a, b, z)
        
        # Term 2: (Γ(b-1) / Γ(a)) * z^(1-b) * ₁F₁(a-b+1; 2-b; z)
        term2 = (gamma(b - 1) / gamma(a)) * z^(1 - b) * _₁F₁(a - b + 1, 2 - b, z)
        
        return term1 + term2
    end
    
end