# use main packages
using DataFrames
using CSV
using Plots
theme(:dark)
using GLM

include("model.jl")
include("functools.jl")
include("preprocessing.jl")

# set data folder
data_folder = "/home/void/Desktop/Data"

function load_data(path::String)::DataFrame
    return CSV.read(data_folder*"/"*path)
end

function main()
    data::DataFrame = load_data("csv/Concrete_Data_Yeh.csv")
    target = "csMPa"
    numerical = ["cement","slag","flyash","water","superplasticizer","coarseaggregate","fineaggregate","age"]
    categorical = String[]
    degree = 1
    include_bias = true
    model = LinearModel(data, target, numerical, categorical, degree, include_bias)
    println(model.regression.coefficients)
end

function test()
    data = DataFrame(A=[1, 2, 3], B=[-1, 0, 1], C=[3, 2, 1])
    println(data)
    x = PolynomialTransformer(data, 2, true)
    println( transform(x, data) )
end

main()
