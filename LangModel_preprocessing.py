import dataset_preprocessing

# get lyrics as nested tokenised list 
# each lyric document would just be 1 of the inner list in the whole collection (training set)
lyrics_list = dataset_preprocessing.musicdf["lyrics"].tolist()
for index in range(len(lyrics_list)):
    lyrics_list[index] = lyrics_list[index].split(" ")

#print(lyrics_list[0:2])
