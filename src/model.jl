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
LinearModel(dataset::DataSet, degree::Int64, include_bias::Bool) = LinearModel(dataset, LinearRegression(dataset, degree, include_bias) )
LinearModel(data::DataFrame, target::String, numerical_features::Array{String,1}, categorical_features::Array{String,1}, degree::Int64, include_bias::Bool) = LinearModel(DataSet(data, target, numerical_features, categorical_features), degree, include_bias)

# predict function
function predict(model::LinearModel, X::DataFrame)::DataFrame
    # get matrix of values times its coefficient
    values = transform( model.regression.polynomial, X) .* model.regression.coefficients
    # return the a Dataframe with the sum of the values per row
    return DataFrame([sum( values[i, :] ) for i in 1:size(values)[1]][:, :], [model.data.target])
end
