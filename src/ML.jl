using DataFrames
using MLDataUtils
using Random

include("model.jl")
include("dataset.jl")
include("functools.jl")

abstract type AbstractML end

#============================================Machine Learning============================================#

struct ML <: AbstractML
    data::DataSet
    train_indices::Array{Float64,1}
    test_indices::Array{Float64,1}
    model_type::DataType
end

function ML(data::DataSet, rate::Float64, model_type::DataType, doshuffle::Bool)::ML
    train, test = train_test_split(data.data, rate, doshuffle)
    return ML(data, train, test, model_type)
end

#============================================Cross Fold============================================#

function KFold(data::DataSet, model_type::DataType, K::UInt8=0x0a, doshuffle::Bool=true, args::Tuple{Vararg{T,N} where N}=())::Float64 where T
    # create base indices
    indices = doshuffle ? shuffle(collect(1:size(data.data)[1])) : collect(1:size(data.data)[1])
    # vectors of scores
    scores = [ score(model_type(DataSet(data.data[train, :], data.target, data.numerical_features, data.categorical_features), args...), data.data[test, :], data.data[test, data.target]) for (train, test) in kfolds(shuffle(indices), K) ]
    # return the mean of the scores
    return mean(scores)
end

function KFold(data::DataFrame, target::String, numerical_features::Array{String,1}, categorical_features::Array{String,1}, model_type::DataType, K::Int64=10, doshuffle::Bool=true, args...)::Float64
    # create base indices
    indices = doshuffle ? shuffle(collect(1:size(data)[1])) : collect(1:size(data)[1])
    # vectors of scores
    scores = [ score(model_type(DataSet(data[train, :], target, numerical_features, categorical_features), args...), data[test, :], data[test, target]) for (train, test) in kfolds(shuffle(indices), K) ]
    # return the mean of the scores
    return mean(scores)
end
