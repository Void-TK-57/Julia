
# use packages
using GLM

# include own files
include("../functools.jl")

abstract type AbstractRegression end
abstract type AbstractLinearRegression <: AbstractRegression end
abstract type AbstractNonLinearRegression <: AbstractRegression end

struct LinearRegression <: AbstractLinearRegression
    degree::Int64
    include_bias::Bool
    coefficients::Array{Float64,1}
end

# function to calculate a linear regression
function LinearRegression(X::Array{T,2}, y::Array{V,1}, degree::Int64=1, include_bias::Bool=true) where {T <: Number, V <: Number}
    # transform the X matrix based on the degree and bias
    X = power_features(convert(Array{Float64,2}, X), degree, include_bias)
    # do a linear regression and get its coefficients
    return LinearRegression(degree, include_bias, coef(lm(X, convert(Array{Float64,1}, y) )) )
end
