import bm25_full
expandedQuery = ["gym", "energy", "hyper", "dance"]
number = 5
selected_docsID, bm25_recommendations = bm25_full.bm25(expandedQuery, n=number)
print(bm25_recommendations)

# check number of relevant songs
import ast
file = open("bm25_relevant.txt", "r")
old_relevant = ast.literal_eval(file.read())
#print(len(old_relevant))