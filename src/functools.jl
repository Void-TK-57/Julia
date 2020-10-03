# use packages
using IterTools
using StatsBase

# function to get all combinations with repetition up to the size passed (and whether it should inclue the empty set)
function combination_with_repetition(array::Array{T,1}, size::Int64, include_empty_set::Bool)::Array{Tuple{Vararg{T,N} where N},1} where {T,N}
    return [chain([filter(issorted, [product(fill(array, d)...)...]) for d in Int(!include_empty_set):size]...)...]
end

# function to get a feature name to the n power
function feature_to_n(str, n)
    if n == 0
        return ""
    elseif n == 1
        return str
    else
        return str*"^"*string(n)
    end
end

# function to replace a value of an array
function replace(vector::Array{T,1}, old_value::T, new_value::T)::Array{T,1} where T
    vector[vector.==old_value] .= new_value
    # return the vector
    return vector
end

# function to do a element wise multiplication
function prod(itr, n::Int64=0, f::Function=x->x)
    # initial value (for empty itr)
    start = ones(n)
    # for each value in itr do multiplication element wise apply the function
    for i in itr
        start = start .* f(i)
    end
    return start
end

#power_features(features, degree, include_bias) = [ length([v...]) == 0 ? "1" : join( filter(x->x!="", [ n_str(features[i], (addcounts!(Dict{Int64,Int64}(), [v...]))[i-1]) for i in 1:length(features) ]), '.')  for v in chain([filter( issorted, Any[product(fill(0:length(features)-1, n)...)...]) for n in Int(!include_bias):degree]...)  ]
