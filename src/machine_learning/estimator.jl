
using DataFrames

struct Estimator
    data::DataFrame
    target::Array{String,1}
    numerical_features::Array{String,1}
    categorical_features::Array{String,1}
end

data::DataFrame, target::Array{String,1}, numerical_features::Array{String,1}, categorical_features::Array{String,1}

fit!(estimator::T,
    data::DataFrame,
    target::Array{String,1},
    numerical_features::Array{String,1},
    categorical_features::Array{String,1} )::T where {T <: Estimator} = estimator

predict(estimator::T, data::DataFrame)::DataFrame where {T <: Estimator} = data

tranform(estimator::T, data::DataFrame)::DataFrame where {T <: Estimator} = predict(estimator, data)

score(estimator::T, data::DataFrame)::Float64 where {T <: Estimator} = 0.0
