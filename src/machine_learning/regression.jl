
# use packages
using GLM

# include own files
include("../functools.jl")

struct Linear_Regression
    coefficients::Array{Float64,1}
end

function linear_regression(X::Array{T,2}, y::Array{V,1}, degree::Int64=1, include_bias::Bool=true)::Array{Float64,1} where {T <: Number, V <: Number}
    # transform the X matrix based on the degree and bias
    X = power_features(convert(Array{Float64,2}, X), degree, include_bias)
    # do a linear regression and get its coefficients
    return coef(lm(X, convert(Array{Float64,1}, y) ))
end
