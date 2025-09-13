import csv
import pandas as pd
import os
import numpy as np

df_j = pd.read_csv(os.path.join("./results", "vote.csv"), encoding='utf-8-sig', dtype={"id": str, "TARGET": str})
df_l = pd.read_csv(os.path.join("./results", "vote_llama.csv"), encoding='utf-8-sig', dtype={"id": str, "TARGET": str})
df_b = pd.read_csv(os.path.join("./results", "vote_gpt.csv"), encoding='utf-8-sig', dtype={"id": str, "TARGET": str})

ids = df_b["id"].values.tolist()
targets = []
df_l_tar = df_l["TARGET"].values.tolist()
df_b_tar = df_b["TARGET"].values.tolist()

short = 0
success = 0
success_1 = 0
success_2 = 0
THRESH = 5
for i, tar in enumerate(df_j["TARGET"].values.tolist()):
    # reference
    l_tar = df_l_tar[i]
    b_tar = df_b_tar[i]
    # check null
    if tar is np.nan:
        if b_tar is not np.nan:
            targets.append(b_tar)
        elif l_tar is not np.nan:
            targets.append(l_tar)
        else:
            targets.append(None)
        continue

    # check too much
    tars = tar.split(",")
    l_tars = []
    b_tars = []
    if len(tars) < THRESH:
        targets.append(tar)
        continue
    
    short += 1
    merged_b = []
    if b_tar is not None:
        b_tars = b_tar.split(",")
        merged_b = list(set(tars) & set(b_tars))
    if len(merged_b) < THRESH and len(merged_b) > 0:
        targets.append(",".join(merged_b))
        success_1 += 1
        continue
    
    merged_l = []
    if l_tar is not None:
        l_tars = l_tar.split(",")
        if len(merged_b) > 0:
            merged_l = list(set(merged_b) & set(l_tars))
        else:
            merged_l = list(set(tars) & set(l_tars))
    if len(merged_l) < THRESH and len(merged_l) > 0:
        targets.append(",".join(merged_l))
        success_2 += 1
        continue

    if len(merged_b) < len(tars) and len(merged_b) > 0:
        targets.append(",".join(merged_b))
        continue
    if len(merged_l) < len(tars) and len(merged_l) > 0:
        targets.append(",".join(merged_l))
        continue
    
    # if len(b_tars) < len(l_tars) and len(b_tars) < len(tars) and len(b_tars) > 0:
    #     targets.append(b_tar)
    # elif len(l_tars) < len(b_tars) and len(l_tars) < len(tars) and len(l_tars) > 0:
    #     targets.append(l_tar)
    # else:
    targets.append(tar)

final_df = pd.DataFrame({'id': ids, 'TARGET': targets})
output_path = 'results/merged.csv'
final_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print("need shorten: ", short)
print("shorten success at stage 1: ", success_1)
print("shorten success at stage 2: ", success_2)

for i, tar in enumerate(targets):
    if tar is None:
        print("empty: ", i)
    elif len(tar.split(",")) < THRESH:
        success += 1
    else:
        print("too long: ", i)
print("shorten success: ", success)

    
    



    
