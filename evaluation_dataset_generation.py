import pandas as pd

# load query csv file and music csv file
musicdf = pd.read_csv('tcc_ceds_music.csv')
querydf = pd.read_csv('evaluation_dataset.csv')

# convert songIDs column in querydf to dtype 'object'
querydf = querydf.astype({'songIDs':object})

for row in range(len(querydf)):
    # find column(s) corresponding to this query
    columns = querydf.loc[row,'relevant_cols']
    col_list = columns.split(",")

    # calculate score of each song by summing the scores in relevant columns
    musicdf['score'] = musicdf.loc[:,col_list].sum(axis=1)

    # sort the songs
    final_df = musicdf.sort_values(by=['score'], ascending=False)

    # return top 30 songs for those attributes
    sorted_song_ids = final_df['doc_id'].to_list()
    final_song_list = sorted_song_ids[:30] # change value here to amount we want to output

    # write relevant docs for this query
    querydf.loc[row,'songIDs'] = final_song_list

# output results as new csv
querydf.to_csv('evaluation_dataset_filled.csv', index=False)
