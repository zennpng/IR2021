import pandas as pd
import numpy as np
from dataset_preprocessing import musicdf

LTR_data = []
eval_df = pd.read_csv('evaluation_dataset_filled.csv') 
eval_df = eval_df.drop(["relevant_cols"], axis=1)
for index, row in eval_df.iterrows():
    query = row["queryString"]
    song_list = row["songIDs"]
    song_list = song_list[1:-1]
    song_list = song_list.split(", ")
    for song in song_list:
        song_lyrics = musicdf.at[int(song), 'lyrics']
        LTR_data.append([query,song_lyrics])

LTR_training_data = pd.DataFrame(LTR_data, columns=['context', 'response'])
#print(LTR_training_data)

#get max length of context and response
measurer = np.vectorize(len)
print(measurer(LTR_training_data.values.astype(str)).max(axis=0))

LTR_training_data.to_csv("LTR_training_data.csv", index=False)


