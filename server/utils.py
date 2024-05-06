from datetime import datetime

def transform_adl_data(top5_df, atomic_df):

   
    data = {}
    for index, row in top5_df.iterrows():
     
        data[f'action_{index + 1}'] = row['Class Name']
        data[f'score_{index + 1}'] = row['Score'].item()
        

    atomic_dict = dict(zip(atomic_df['Class Name'], atomic_df['Score']))


    # -- replace with a loop
    data['sit_score'] = atomic_dict['sit'].item()
    data['lie_score'] = atomic_dict['lie'].item()
    data['walk_score'] = atomic_dict['walk'].item()
    data['stand_score'] = atomic_dict['stand'].item()
    data['video_name'] = 'example.mp4'
    data['timestamp'] = datetime.now()

    return data
     