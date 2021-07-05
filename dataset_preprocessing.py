import pandas as pd
import matplotlib.pyplot as plt

# remove irrelevant columns
musicdf = pd.read_csv('tcc_ceds_music.csv')
musicdf = musicdf.drop(['len', 'age'], axis=1)
#print(musicdf)

# column tagging
mood_attributes = ["dating", 
                   "violence",
                   "world/life",  # neglect
                   "night/time",
                   "shake the audience",  # neglect
                   "family/gospel",
                   "romantic",
                   "communication",  # neglect
                   "obscene",
                   "music",  # neglect
                   "movement/places",
                   "light/visual perceptions",  # neglect
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

# 1. No. of songs
'''
28372
'''

# 2. most common artists
#print(musicdf['artist_name'].value_counts()[0:5]) 
'''
johnny cash        190
ella fitzgerald    188
dean martin        146
willie nelson      131
george jones       107
'''

# 3. year breakdown
#print(musicdf['release_date'].value_counts())
'''
2017    660
2018    653
2015    617
2009    597
2012    588
'''

# 4. year frequency plot
fig, ax = plt.subplots(figsize=(20, 5))
musicdf['release_date'].value_counts().sort_index().plot(ax=ax, kind='bar')
#fig.savefig("year_freq_plot.jpg")


# 5. genre breakdown 
#print(musicdf['genre'].value_counts())
'''
pop        7042
country    5445
blues      4604
rock       4034
jazz       3845
reggae     2498
hip hop     904
'''

# 6. song name breakdown
#print(musicdf['track_name'].value_counts()[0:5])
'''
tonight             17
hold on             15
stay                15
without you         14
yesterday           13
'''
#print(musicdf['track_name'].nunique())
'''
23687
'''

# get statistics about attributes in the dataset

# 1. sorted mean of mood and quality attributes
#print(musicdf[mood_attributes].describe().loc[['mean','min','max']].T.sort_values(by='mean', ascending=False))
'''
                            mean       min       max
sadness                   0.129389  0.000284  0.981424
world/life                0.120973  0.000291  0.962105
violence                  0.118396  0.000284  0.981781
obscene                   0.097168  0.000289  0.992298
communication             0.076680  0.000291  0.645829
music                     0.060047  0.000289  0.956938
night/time                0.057387  0.000289  0.973684
light/visual perceptions  0.049005  0.000284  0.667782
romantic                  0.048681  0.000284  0.940789
movement/places           0.047401  0.000284  0.638021
feelings                  0.030996  0.000289  0.958810
like/girls                0.028057  0.000284  0.594459
family/spiritual          0.024150  0.000284  0.618073
dating                    0.021112  0.000291  0.647706
shake the audience        0.017422  0.000284  0.497463
family/gospel             0.017045  0.000289  0.545303
'''
#print(musicdf[quality_attributes].describe().loc[['mean','min','max']].T.sort_values(by='mean', ascending=False))
'''
                      mean           min       max
loudness          0.665249  0.000000e+00  1.000000
energy            0.569875  0.000000e+00  1.000000
danceability      0.533348  5.415358e-03  0.993502
valence           0.532864  0.000000e+00  1.000000
acousticness      0.339235  2.810000e-07  1.000000
instrumentalness  0.080049  0.000000e+00  0.996964
'''

# 2. topic breakdown
topic_breakdown = musicdf['topic'].value_counts()
#print(topic_breakdown)  # only 8 out of the 16 moods are dominant in songs 
'''
sadness       6096
violence      5710
world/life    5420
obscene       4882
music         2303
night/time    1825
romantic      1524
feelings       612
'''
