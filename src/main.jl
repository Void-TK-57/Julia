# use main packages
using DataFrames
using CSV
using Plots
using GLM

include("model.jl")
include("functools.jl")
include("preprocessing.jl")
include("ML.jl")

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
    plt = scatter(y_values[target], y_predict[target], markercolor = :lime, markeralpha = 0.3)
    line = [min(y_values[target]...), max(y_values[target]...)]
    plot!(line, line, linecolor=:red, linewidth = 3)
    display(plt)

    println( "Explained Variance: ", 100.0*explained_variance(y_values[target], y_predict[target]), "%" )
    # do k fold
    score_k_fold = KFold(data, target, numerical, categorical, LinearModel, 10, true, 2, false)
    println( "Explained Variance: ", 100.0*score_k_fold, "%" )
end

function test()
    data = DataFrame(A=[1, 2, 3, 4, 5, 6, 7], B=[-1, 0, 1, 7, 1, 5, 0], C=[3, 2, 1, 1, 23, -2, 5])
    println(data)
    test, train = train_test_split(data, 0.25, false)
    println(data[test , :])
    println(data[train, :])
end

main()
