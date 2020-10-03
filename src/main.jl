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
    degree = 3
    include_bias = true
    model = LinearModel(data, target, numerical, categorical, degree, include_bias)
    p = predict(model, data)
    plt = scatter(data[:csMPa], p)
    # construct a identity line
    interval = [min(p...), max(p...)]
    plot!(interval, interval)
    display(plt)
end

function test()
    data = DataFrame(A=[1, 2, 3, 4, 5], B=[-1, 2, 1, -2, 0], C=[0, 0, -1, 0, 4])
    println(data)
    println(transform(MinMaxTransformer(data), data))
end

test()
