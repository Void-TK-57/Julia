# use main packages
using DataFrames
using CSV
using Plots
theme(:dark)
using GLM

include("machine_learning/model.jl")
include("functools.jl")

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
    sample = 1:10
    println(sample)
    scale = min_max_scale(sample[:, :])[:]
    println(scale)
end

test()
