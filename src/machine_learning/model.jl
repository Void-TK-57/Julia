
include("regression.jl")

abstract type AbstractModel end
abstract type AbstractLinearModel <: AbstaractModel end
abstract type AbstractNonLinearModel <: AbstractModel end

# base dataset of a model
struct Dataset
    data::DataFrame
    target::Array{String,1}
    numerical_features::Array{String,1}
    categorical_features::Array{String,1}
end

#data::DataFrame, target::Array{String,1}, numerical_features::Array{String,1}, categorical_features::Array{String,1}

#============================================Linear Model============================================#

struct Linear_Model <: AbstractLinearModel
    data::Dataset
    degree::Int64
    include_bias::Bool
    coefficients::Array{Float64,1}
end

# constructor with a data to be fit
Linear_Model(data::Dataset, degree::Int64, include_bias::Bool) = Linear_Model(data, degree, include_bias, linear_regression() )
Linear_Model(data::DataFrame, target::Array{String,1}, numerical_features::Array{String,1}, categorical_features::Array{String,1}, degree::Int64, include_bias::Bool) = Linear_Model(Dataset(data, target, numerical_features, categorical_features), degree, include_bias, linear_regression() )
