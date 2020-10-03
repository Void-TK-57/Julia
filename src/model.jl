# use packages
using LinearAlgebra

# include own filesinclude("regression.jl")
include("dataset.jl")
include("functools.jl")
include("regression.jl")

abstract type AbstractModel end
abstract type AbstractLinearModel <: AbstractModel end
abstract type AbstractNonLinearModel <: AbstractModel end


#data::DataFrame, target::Array{String,1}, numerical_features::Array{String,1}, categorical_features::Array{String,1}

#============================================Linear Model============================================#

struct LinearModel <: AbstractLinearModel
    data::DataSet
    regression::LinearRegression
end

# constructor with a data to be fit
LinearModel(dataset::DataSet, degree::Int64, include_bias::Bool) = LinearModel(dataset, LinearRegression(Matrix(dataset.data[:, dataset.numerical_features]), dataset.data[:, dataset.target], degree, include_bias) )
LinearModel(data::DataFrame, target::String, numerical_features::Array{String,1}, categorical_features::Array{String,1}, degree::Int64, include_bias::Bool) = LinearModel(DataSet(data, target, numerical_features, categorical_features), LinearRegression(Matrix(data[:, numerical_features]), data[:, target], degree, include_bias) )


# predict function
predict(model::LinearModel, X::DataFrame)::Array{Float64,1} = ( power_features(convert(Array{Float64,2}, X[:, model.data.numerical_features]), model.regression.degree, model.regression.include_bias)*model.regression.coefficients[:, :] )[:]
