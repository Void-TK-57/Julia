# use packages
using DataFrames
using StatsBase
using Statistics

# include own files
include("functools.jl")

abstract type AbstractTransformer end

#=========================== Min Max Transformer ===========================#

struct MinMaxTransformer <: AbstractTransformer
    minimum_values::DataFrame
    maximum_values::DataFrame
end

function MinMaxTransformer(X::DataFrame)::MinMaxTransformer
    return MinMaxTransformer( mapcols(x->min(x...), X), mapcols(x->max(x...), X) )
end

function transform(transformer::MinMaxTransformer, X::DataFrame)::DataFrame
    return (X .- transformer.minimum_values) ./ (transformer.maximum_values .- transformer.minimum_values)
end


#=========================== Mean Std Transformer ===========================#

struct MeanStdTransformer <: AbstractTransformer
    mean_values::DataFrame
    std_values::DataFrame
end

function MeanStdTransformer(X::DataFrame)
    # calculate std and mean together for efficiency
    mean_and_std_values = mapcols(mean_and_std, a)
    return MeanStdTransformer( mapcols(x->[ms[1] for ms in x], mean_and_std_values), mapcols(x->[ms[2] for ms in x], mean_and_std_values) )
end

function transform(transformer::MeanStdTransformer, X::DataFrame)::DataFrame
    return (X .- transformer.mean_values) ./ transformer.std_values
end

#=========================== Polynomial Transformer ===========================#

struct PolynomialTransformer <: AbstractTransformer
    degree::Int64
    include_bias::Bool
    names_combination::Array{Tuple{Vararg{String,N} where N},1}
    names::Array{String,1}
    names_combined::Array{String,1}
end

function PolynomialTransformer(X::DataFrame, degree::Int64, include_bias::Bool)
    # get combinations, the features and then the features combined
    combinations = combination_with_repetition(names(X), degree, include_bias)
    features = names(X)
    features_combined = replace([ join( filter(x->x!="", [feature_to_n(i, count(x->x==i,name_combination)) for i in features]), '.')  for name_combination in combinations], "", "1")
    # call default constructor with the values calculated
    return PolynomialTransformer(degree, include_bias, combinations, features, features_combined)
end

function transform(transformer::PolynomialTransformer, X::DataFrame)::DataFrame
    return rename(DataFrame([prod(name_combination, size(X)[1], i->X[i]) for name_combination in transformer.names_combination]), transformer.names_combined)
end

#=========================== Imputer Transformer ===========================#

struct Imputer <: AbstractTransformer
    missing_values
    strategy::String
    fill_value
end

function Imputer(X::DataFrame, missing_values, strategy::String, fill_value)
    return Imputer(missing_values, strategy, fill_value)
end

function transform(transformer::Imputer, X::DataFrame)::DataFrame
    # check strategy
    if transformer.strategy == "mean"
        f = v->mean(filter(x->x!==NaN, v))
    elseif transformer.strategy == "median"
        f = v->median(filter(x->x!==NaN, v))
    else
        f = v->transformer.fill_value
    end
    # for each column replace missing value by the value from f
    return mapcols(x->replace(x, transformer.missing_values, f(x)), X)
end

#=========================== Pipeline Transformer ===========================#

struct Pipeline <: AbstractTransformer
    transformers::Array{T,1} where {T <: AbstractTransformer}
end

function Pipeline(X::DataFrame, transformers_types::Array{DataType,1}, args::Array{Tuple{Vararg{T,N} where N},1}) where {T <: Any}
    # array of transformers
    transformers = Array{AbstractTransformer,1}()
    # for each value in the array of transformers
    for i in 1:length(transformers_types)
        # create the transformer with its args, push it to the vector, and call transform on X
        transformer = transformers_types[i](X, args[i]...)
        push!(transformers, transformer)
        X = transform(transformer, X)
    end
    # call default constructor
    return Pipeline(transformers)
end

function transform(pipe::Pipeline, X::DataFrame)::DataFrame
    # for each transformer in the pipeline, apply the transform
    for transformer in pipe.transformers
        X = transform(transformer, X)
    end
    # return the final X calculated
    return X
end
