# use main packages
using DataFrames
using CSV
using Plots
using GLM

include("machine_learning/regression.jl")
include("functools.jl")

# set data folder
data_folder = "/home/void/Desktop/Data"

function load_data(path::String)::DataFrame
    return CSV.read(data_folder*"/"*path)
end

function main()
    data::DataFrame = load_data("csv/linear/train.csv")
    data[:y] = data[:y].+50
    println(head(data))
    x = Matrix(data[[:x]])
    model = lm(power_features(x, 1, true), data[:y])
    println(model)
    println(coef(model))
    display( scatter(data[:x], data[:y]) )

end

main()
