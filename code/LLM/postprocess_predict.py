import pandas as pd
import glob
import json

"""
Voting Part
"""
file_path_pattern = 'result/checkpoint_105/checkpoint_*.csv'
output_file = 'result/checkpoint_105/vote_checkpoint_105_20.csv'
reference_file = 'result/checkpoint_105/checkpoint_105_0.csv'

all_files = glob.glob(file_path_pattern)
dataframes = []
for file in all_files:
    df = pd.read_csv(file)
    dataframes.append(df)
merged_df = pd.concat(dataframes, ignore_index=True)

expanded_df = merged_df.assign(TARGET=merged_df['TARGET'].str.split(",")).explode('TARGET')
expanded_df['TARGET'] = expanded_df['TARGET'].str.strip()

record_counts = expanded_df.groupby(['id', 'TARGET']).size().reset_index(name='count')
filtered_df = record_counts[record_counts['count'] >11] # 20次裡面投了12次以上

vote_df = df.groupby('id', as_index=False).agg({
    'TARGET': lambda x: ','.join(x)
})

reference_df = pd.read_csv(reference_file)
vote_df = vote_df.set_index('id').reindex(reference_df['id']).reset_index()
vote_df.to_csv(output_file, index=False, encoding='utf-8-sig')

"""
Check Law list
"""
law_file = 'data/law_list.json'
filtered_file = 'result/checkpoint_105/vote_checkpoint_105_20_filtered.csv'

with open(law_file, 'r', encoding='utf-8') as f:
    law_ids = json.load(f)

submission_df = vote_df

def filter_valid_targets(targets):
    targets_list = [target.strip() for target in targets.split(',') if target.strip()]
    valid_targets = [target for target in targets_list if target in law_ids]
    invalid_targets = [target for target in targets_list if target not in law_ids]
    print(invalid_targets)
    return ','.join(valid_targets)

submission_df['TARGET'] = submission_df['TARGET'].apply(filter_valid_targets)

submission_df[['id', 'TARGET']].to_csv(filtered_file, index=False, encoding='utf-8')
