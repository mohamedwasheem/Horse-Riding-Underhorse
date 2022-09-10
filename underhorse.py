# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 18:04:58 2022

@author: User
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
df_og = pd.read_csv('nyra_2019_complete_proper.csv')

## This section has some pre processing details, 
## data is large so i ran it once and saved the output for future use
#################

# old_cols = ['track_id','race_date','race_number','program_number','trakus_index',
# 'latitude','longitude','distance_id','course_type','track_condition','run_up_distance',
# 'race_type','purse','post_time','weight_carried','jockey','odds','position_at_finsh']

# df_og = pd.read_csv('nyra_2019_complete.csv')
# initial_line = list(df_og.columns)
# df_og.columns = old_cols
# df_og.loc[-1] = initial_line
# df_og.index = df_og.index + 1
# df_og.sort_index(inplace=True)
# df_og.to_csv('nyra_2019_complete_proper.csv',index=False)

# These are the cols present in the df
# ['track_id', 'race_date', 'race_number', 'program_number',
#        'trakus_index', 'latitude', 'longitude', 'distance_id', 'course_type',
#        'track_condition', 'run_up_distance', 'race_type', 'purse', 'post_time',
#        'weight_carried', 'jockey', 'odds', 'position_at_finsh']

# Lets treat try to group the features

# Descriptive features
# 'track_id', 'race_date', 'race_number', 'program_number', 'post_time'
# Distance features
# 'trakus_index', 'latitude', 'longitude', 'distance_id',
# Track features
# 'course_type', 'track_condition', 'run_up_distance', 'race_type'
# Jockey features
# 'purse','weight_carried','jockey', 
# Winning features
# 'odds', 'position_at_finsh'

## Lets take a look at odds and position finished

cols2sel = ['track_id', 'race_date', 'race_number', 'program_number', 'odds', 'position_at_finsh']
df_odds = df_og[cols2sel]

## This function will provide the number of under dog wins
def get_under_dog(df):
    if len(df.drop_duplicates('odds')) != len(df['program_number'].unique()):
        print("TIE!!")
    df_x = df.drop_duplicates('program_number')
    odds = np.asarray(df_x.odds.unique() )
    odds = np.flip(np.sort(odds))
    pos_list = []
    pos_counter = 1
    for odd in odds:
        for i in range(0,len(df_x[df_x['odds']==odd])):
            pos_list.append(pos_counter)
        pos_counter = pos_counter + 1
        
    tie_pos = df_x.value_counts('position_at_finsh') > 1
    ## Based on the odds lets create a expected positon list
    
    df_x.sort_values('odds',ascending=False,inplace=True)
    # position_list = list(np.arange(1,len(df_x)+1))
    position_list = list(pos_list)
    df_x['Exp_pos'] = position_list
    df_x['Exp_final'] = df_x['Exp_pos'] - df_x['position_at_finsh']
    under_dog_wins = len(df_x[df_x['Exp_final']>0])
    total_horses = len(df_x)
    return under_dog_wins,total_horses

df_gb = df_odds.groupby(['track_id','race_date','race_number'])

track_list = []
race_date_list = []
race_num_list = []
under_dog_list = []
total_horse_list = []
for key,vals in df_gb:
    # print(key)
    under_dog_wins,total_horses = get_under_dog(vals)
    under_dog_list.append(under_dog_wins)
    total_horse_list.append(total_horses)
    track_list.append(key[0])
    race_date_list.append(key[1])
    race_num_list.append(key[2])

odds_analysis = pd.DataFrame([track_list,race_date_list,race_num_list,total_horse_list,under_dog_list])
odds_analysis = odds_analysis.transpose()
odds_analysis.columns = ['track_id', 'race_date', 'race_number','total_horses','num_underdog_wins']
odds_analysis['total_horses'] = odds_analysis['total_horses'].astype(int)
odds_analysis['num_underdog_wins'] = odds_analysis['num_underdog_wins'].astype(int)
    
underdog_win_perc =     (odds_analysis['num_underdog_wins'].sum() / odds_analysis['total_horses'].sum()) * 100

plt.figure()
sns.barplot(['Total Horses','Total Under-Horse wins'],[odds_analysis['total_horses'].sum(),odds_analysis['num_underdog_wins'].sum()],palette='bright')
plt.title('Against Odds Wins')
plt.savefig('Agains Odds Wins.png')
