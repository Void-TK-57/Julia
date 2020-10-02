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
    data[:y] = data[:y].+20
    println(head(data))
    degree = 2
    include_bias = true
    model = linear_regression(Matrix(data[[:x]]), data[:y], degree, include_bias)
    println(model)
    coefs = model.coefficients
    println(rename(DataFrame(Matrix( transpose( coefs ) ) ), features_names(["X"], degree, include_bias) ))
    display( scatter(data[:x], data[:y]) )

end

main()
