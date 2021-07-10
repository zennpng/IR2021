import bm25_full
expandedQuery = ["workout", "energy", "hyper", "dance"]
number = 5
selected_docsID, bm25_recommendations = bm25_full.bm25(expandedQuery, n=number)
print(bm25_recommendations)