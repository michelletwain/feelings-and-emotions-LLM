import pandas as pd

# read the partial files
df1 = pd.read_csv("llama_panas_1_to_200.csv")
df2 = pd.read_csv("llama_panas_201_to_end.csv")

# concatenate rows and reset the row index
merged = pd.concat([df1, df2], ignore_index=True)

# (optional) drop any exact-duplicate rows that might slip in
# merged = merged.drop_duplicates()

# save the result
merged.to_csv("llama_panas_full.csv", index=False)
print("Merged file written to llama_panas_full.csv")
