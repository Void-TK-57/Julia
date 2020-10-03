using DataFrames
include("../functools.jl")

abstract type AbstractPreprocessing end

# function to do a min max scaling
function min_max_scale(X::Array{T,2}, min_interval::Int64=0, max_interval::Int64=1)::Array{T,2} where {T <: Number}
    # transpose first
    Xt = transpose(X)
    # calcualte minimum and maximum values per column
    min_ = min_cols(Xt)
    max_ = max_cols(Xt)
    # scale it by the max and min values, and the to the ranges passed
    Xt_scaled = (Xt.-min_) ./ ( (Xt.+max_) - (Xt.-min_) )
    Xt_ranged = ( Xt_scaled.+min_interval ) .* ( max_interval - min_interval )
    # return the trapose of the Ranged Scaled Matrix
    return transpose(Xt_ranged)
end
