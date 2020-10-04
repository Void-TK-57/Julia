
# use packages
using GLM
using DataFrames

# include own files
include("functools.jl")
include("dataset.jl")
include("preprocessing.jl")

abstract type AbstractRegression end
abstract type AbstractLinearRegression <: AbstractRegression end
abstract type AbstractNonLinearRegression <: AbstractRegression end

struct LinearRegression <: AbstractLinearRegression
    polynomial::PolynomialTransformer
    coefficients::DataFrame
end

# function to calculate a linear regression
function LinearRegression(data::DataSet, degree::Int64=1, include_bias::Bool=true)
    # create a polynomial transformer
    poly=PolynomialTransformer(data.data[data.numerical_features], degree, include_bias)
    # get X matrix expanded for the regression
    X = Matrix(transform(poly, data.data[data.numerical_features]))
    # do a linear regression and get its coefficients
    return LinearRegression(poly, rename(DataFrame( Matrix(transpose(coef(lm(X, convert(Array{Float64,1}, data.data[data.target]))) )) ), poly.names_combined) )
end
