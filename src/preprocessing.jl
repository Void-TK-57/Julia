# use packages
using DataFrames
using StatsBase

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
