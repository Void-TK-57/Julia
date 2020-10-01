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
