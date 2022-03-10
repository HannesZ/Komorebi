def convert_categoricals(data_frame, cols):
    col_list = [c for c in data_frame.columns if (data_frame[c].dtype.name != 'category' and c in cols)]
    if len(col_list) > 0:
        print('converting categoricals...')
        for i, c in enumerate(col_list):
            data_frame[c] = data_frame[c].astype('category')
            print(f".................. " + c + ":  " + '{:.1%}'.format((i+1)/len(col_list)))
        return data_frame