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

#=========================== Polynomial Transformer ===========================#

struct PolynomialTransformer <: AbstractTransformer
    degree::Int64
    include_bias::Bool
    names_combination::Array{Tuple{Vararg{String,N} where N},1}
    names::Array{String,1}
end

function PolynomialTransformer(X::DataFrame, degree::Int64, include_bias::Bool)
    return PolynomialTransformer(degree, include_bias, combination_with_repetition(names(X), degree, include_bias), names(X))
end

function transform(transformer::PolynomialTransformer, X::DataFrame)
    return rename(DataFrame([prod(name_combination, size(X)[1], i->X[i]) for name_combination in transformer.names_combination]), replace([ join( filter(x->x!="", [feature_to_n(i, count(x->x==i,name_combination)) for i in transformer.names]), '.')  for name_combination in transformer.names_combination], "", "1"))
end
