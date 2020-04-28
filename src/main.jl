# use pandas library
using Pandas

# get input file
input_file = ARGS[1]
# read input file
df = read_csv(input_file)
println(df)
println(iloc(df)[1])
