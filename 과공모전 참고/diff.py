import pandas as pd
import numpy as np

whisper = pd.read_csv('result_submit.csv', encoding='euc-kr')
my = pd.read_csv('result_mine.csv', encoding='euc-kr')

merged_df = pd.merge(whisper, my, on=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7'], suffixes=('_whisper', '_my'))
merged_df['diff'] = merged_df['V8_whisper'] - merged_df['V8_my']
merged_df['times'] = merged_df['V8_whisper'] / merged_df['V8_my']
merged_df['times'].replace([np.inf, -np.inf], np.nan, inplace=True)
merged_df = merged_df.sort_values(by='diff', ascending=False)
print(merged_df)
merged_df.to_csv('letssee.csv', index=False, encoding='euc-kr')