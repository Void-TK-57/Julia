using IterTools

# combinations of indices function
comb(n_features, degree, include_bias) = chain([filter(issorted, Any[product(fill(0:n_features-1, n)...)...]) for n in Int(!include_bias):degree]...)

function power_features(X::AbstractArray{T}, degree::Int64, include_bias::Bool=false) where T
    # get number of sample and features
    n_samples, n_features = size(X)
    # calculate features combinations
    combinations = comb(n_features, degree, include_bias)
    # count output
    output_size = count(_->true, combinations)
    # allocate memory for the new matrix
    X_ = Matrix{T}(undef, n_samples, output_size)
    # for each combination,
    for (i, c) in enumerate(combinations)
        # set the column to the product
        X_[:, i] = prod(X[:, Int[k+1 for k in c]], dims=2)
    end
    return X_
end

# feature name function
function n_str(str, n)
    if n == 0
        return ""
    elseif n == 1
        return str
    else
        return str*"^"*string(n)
    end
end



function features_names(features, degree, include_bias)
    # empty dict function
    empty_dict() = Dict{Int64,Int64}()
    # function to get feature string based on the map
    calculate_str(map_combination) = join( filter(x->x!="", [ n_str(features[i], map_combination[i-1]) for i in 1:length(features) ]), '.')
    # get features based on the combinations
    features = [ calculate_str( addcounts!(empty_dict(), [v...]) )  for v in comb(length(features), degree, include_bias)] ]
    # return the features
    return features
end


# terror_function(features, degree, include_bias) = [ join( filter(x->x!="", [ n_str(features[i], (addcounts!(Dict{Int64,Int64}(), [v...]))[i-1]) for i in 1:length(features) ]), '.')  for v in (chain([filter( issorted, Any[product(fill(0:length(features)-1, n)...)...]) for n in Int(!include_bias):degree]...) ) ] ]
