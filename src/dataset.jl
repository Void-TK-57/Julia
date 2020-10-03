# use packages
using DataFrames

struct DataSet
    data::DataFrame
    target::String
    numerical_features::Array{String,1}
    categorical_features::Array{String,1}
end
