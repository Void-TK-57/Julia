# use main packages
using DataFrames
using CSV
using Plots
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
    include_bias = false
    model = LinearModel(data, target, numerical, categorical, degree, include_bias)
    y_values = model.data.data[[model.data.target]]
    y_predict = predict(model, model.data.data[model.data.numerical_features])
    println(head( y_values  ))
    println(head( y_predict ))
    plt = scatter(y_values[target], y_predict[target], markercolor = :lime, markeralpha = 0.3)
    line = [min(y_values[target]...), max(y_values[target]...)]
    plot!(line, line, linecolor=:red, linewidth = 3)
    display(plt)
end

function test()
    data = DataFrame(A=[1, 2, 3], B=[-1, 0, 1], C=[3, 2, 1])
    println(data)
    x = PolynomialTransformer(data, 2, true)
    println( transform(x, data) )
    # create pipeline line = (:steppre, :dot, :arrow, 0.5, 4, :red)
    pipeline = Pipeline(data, [PolynomialTransformer, MinMaxTransformer, Imputer], [(2, true), (), (NaN, "fill", 1.0)])
    final = transform(pipeline, data)
    println(final)
end

main()
