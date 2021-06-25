import pandas as pd

# remove irrelevant columns
musicdf = pd.read_csv('tcc_ceds_music.csv')
musicdf = musicdf.drop(['len', 'age'], axis=1)
#print(musicdf)

# column tagging
mood_attributes = ["dating", 
                   "violence",
                   "world/life",
                   "night/time",
                   "shake the audience",
                   "family/gospel",
                   "romantic",
                   "communication",
                   "obscene",
                   "music",
                   "movement/places",
                   "light/visual perceptions",
                   "family/spiritual",
                   "like/girls",
                   "sadness",
                   "feelings"]   # total = 16

quality_attributes = ["danceability",
                      "loudness",
                      "acousticness",
                      "instrumentalness",
                      "valence",
                      "energy"]   # total = 6

song_attributes = ["artist_name",
                   "track_name",
                   "release_date",
                   "genre"]  # total = 4

# get statistics about the songs in the dataset

# 1. No. of songs --> 28372

# 2. most popular artists
print(musicdf['artist_name'].value_counts()[0:5])

# 3. year breakdown
print(musicdf['release_date'].value_counts())

# 4. genre breakdown 
print(musicdf['genre'].value_counts())

# 5. song name breakdown
print(musicdf['track_name'].value_counts()[0:10])
print(musicdf['track_name'].nunique())

# get statistics about attributes in the dataset

# 1. sorted mean of mood and quality attributes
print(musicdf[mood_attributes].describe().loc[['mean','min','max']].T.sort_values(by='mean', ascending=False))
print(musicdf[quality_attributes].describe().loc[['mean','min','max']].T.sort_values(by='mean', ascending=False))

# 2. topic breakdown
topic_breakdown = musicdf['topic'].value_counts()
print(topic_breakdown)  # only 8 out of the 16 moods are dominant in songs 

