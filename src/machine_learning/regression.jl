
# use packages
using GLM

# include own files
include("../functools.jl")

struct Linear_Regression
    coefficients::Array{Float64,1}
    degree::Int64
    include_bias::Bool
end

# function to calculate a linear regression
function linear_regression(X::Array{T,2}, y::Array{V,1}, degree::Int64=1, include_bias::Bool=true)::Linear_Regression where {T <: Number, V <: Number}
    # transform the X matrix based on the degree and bias
    X = power_features(convert(Array{Float64,2}, X), degree, include_bias)
    # do a linear regression and get its coefficients
    return Linear_Regression(coef(lm(X, convert(Array{Float64,1}, y) )), degree, include_bias)
end

# function to get the features names based on the Regression Model
features_names(model::Linear_Regression, names::Array{AbstractString,1}) = features_names(names, model.degree, model.include_bias)
