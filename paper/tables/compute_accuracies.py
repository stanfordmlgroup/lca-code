import argparse 
import pandas as pd
import math
import os 

def wilson(p_hat, n):
    z = 1.96
    base = (p_hat + (z**2)/(2*n)) / (1 + (z**2)/n)
    diff = (z / (1 + (z**2)/n))*math.sqrt((p_hat*(1-p_hat)/n) + (z**2)/(4*n**2))
    return base-diff, base+diff

def statistics(df):
    p = df.accuracy.mean()
    n = len(df)
    ci = wilson(p, n)
    return p, ci[0], ci[1], df.accuracy.sum()

def results(df):
    experience_df = pd.DataFrame(columns=['unassisted', 'assisted'], index=['specialist', 'nonGI', 'trainee', 'NOC', 'overall'])
    experience_map = {'specialist':1, 'nonGI': 2, 'trainee':3, 'NOC':4}
    assisted_df_big = df.loc[df.loc[:, 'mode'] == 1, :]
    unassisted_df_big = df.loc[df.loc[:, 'mode'] == 0, :]

    experience_df.loc['overall', 'assisted'] = statistics(assisted_df_big)
    experience_df.loc['overall', 'unassisted'] = statistics(unassisted_df_big)
    
    for experience in experience_map.keys():
        experience_num = experience_map[experience]
        unassisted_df = unassisted_df_big.loc[unassisted_df_big.loc[:, 'type'] == experience_num, :]
        experience_df.loc[experience, 'unassisted'] = statistics(unassisted_df) 
        assisted_df = assisted_df_big.loc[assisted_df_big.loc[:, 'type'] == experience_num, :]
        experience_df.loc[experience, 'assisted'] = statistics(assisted_df) 

    grade_df = pd.DataFrame(columns=['unassisted', 'model_correct', 'model_incorrect'], index=['grade3', 'notgrade3'])

    grade_df.loc['grade3', 'unassisted'] = statistics(unassisted_df_big.loc[unassisted_df_big.grade == 3, :])
    grade_df.loc['notgrade3', 'unassisted'] = statistics(unassisted_df_big.loc[unassisted_df_big.grade != 3, :])

    model_correct_df = assisted_df_big.loc[assisted_df_big.errormodel == 0, :]
    grade_df.loc['grade3', 'model_correct'] = statistics(model_correct_df.loc[model_correct_df.grade == 3, :]) 
    grade_df.loc['notgrade3', 'model_correct'] = statistics(model_correct_df.loc[model_correct_df.grade != 3, :])

    model_incorrect_df = assisted_df_big.loc[assisted_df_big.errormodel == 1, :]
    grade_df.loc['grade3', 'model_incorrect'] = statistics(model_incorrect_df.loc[model_incorrect_df.grade == 3, :])
    grade_df.loc['notgrade3', 'model_incorrect'] = statistics(model_incorrect_df.loc[model_incorrect_df.grade != 3, :])

    return experience_df, grade_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--save_dir', type=str)

    args = parser.parse_args()

    df = pd.read_csv(args.data, index_col=0)
    experience_df, grade_df = results(df)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    experience_df.to_csv(os.path.join(args.save_dir, 'experience_accuracies.csv'))
    grade_df.to_csv(os.path.join(args.save_dir, 'grade_accuracies.csv'))

    

