import pandas as pd
import ast
from sklearn.model_selection import train_test_split

evaluation_df = pd.read_csv("evaluation_dataset_filled.csv")
validation, test = train_test_split(evaluation_df, test_size=0.3, random_state=10)

relevantDocs_raw = []
for id_str in validation["songIDs"]:
    relevantDocs_raw += ast.literal_eval(id_str)
relevantDocs = list(set(relevantDocs_raw))

#print(relevantDocs)
#print(len(relevantDocs))