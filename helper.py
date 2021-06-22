# Helper functions to read and preprocess data files from Matlab format
# Data science libraries
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Others
from pathlib import Path
from tqdm.auto import tqdm
import requests

def matfile_to_dic(folder_path):
    '''
    Read all the matlab files of the Paderborn Dataset and return a
    dictionary. The key of each item is the filename and the value is the data
    of one matlab file, which also has key value pairs.

    Parameter:
        folder_path:
            Path (Path object) of the folder which contains the matlab files.
    Return:
        output_dic:
            Dictionary which contains data of all files in the folder_path.
    '''
    output_dic = {}
    for _, filepath in enumerate(folder_path.glob('*.mat')):
        # strip the folder path and get the filename only.
        key_name = str(filepath).split('\\')[-1]
        output_dic[key_name] = scipy.io.loadmat(filepath)

    for _, values in output_dic.items():
        del values['__header__']
        del values['__version__']
        del values['__globals__']

    data_dic = {}
    for _,v in output_dic.items(): # file names, dict
        for k1,v1 in v.items(): # data stored in file name e.g. 'N09_M07_F10_K001_1', dict
            data_dic[k1] = v1

    data_dic_extract = {}
    for k,v in data_dic.items(): #[0][0][2][0]
        for v1 in v: # numpy.ndarray
            for v2 in v1:
                for i,v3 in enumerate(v2):
                    if i == 2:
                        data_dic_extract[k] = (v3)

    data_dic_extract_2 = {}
    for k,v in data_dic_extract.items(): # dic len 80, e.g KEY = 'N09_M07_F10_K001_1', VALUE = force,phase_current_1,etc
        data_dic_temp = {}
        data_cat = ''
        data_num = 0
        for v1 in v: # array of VALUE, len = 1
            for v2 in v1: # array of VALUE, len = 7
                for i,v3 in enumerate(v2): #numpy.void len 16
                    if i == 0:
                        for v4 in v3: #numpy.ndarray len 1, Data types = force, etc.
                            data_cat = str(v4)
                    if i == 2:
                        for v4 in v3: #numpy.ndarray len 1, value of Data types
                            data_num = v4
                data_dic_temp[data_cat] = data_num
        data_dic_extract_2[k] = data_dic_temp

    return data_dic_extract_2

def label(filename):
    '''
    Function to create label for each signal based on the filename. Apply this
    to the "filename" column of the DataFrame.
    Usage:
        df['label'] = df['filename'].apply(label)
    '''
    if 'K00' in filename:
        return 'NORMAL'
    elif 'KI' in filename:
        return 'IR'
    elif 'KA' in filename:
        return 'OR'
    elif 'KB' in filename:
        return 'OR + IR'


def matfile_to_df(folder_path, data_cat):
    '''
    Read all the matlab files in the folder, preprocess, and return a DataFrame
    with data specified by data_cat

    Parameter:
        folder_path:
            Path (Path object) of the folder which contains the matlab files.\
        data_cat:
            Data category of interest, i.e. 'force', 'phase_current_1', 'phase_current_2',
                                            'speed', 'temp_2_bearing_module', 'torque',
                                            'vibration_1'
    Return:
        DataFrame with preprocessed data
    '''
    dic = matfile_to_dic(folder_path)
    df = pd.DataFrame.from_dict(dic).T
    df = df.reset_index().rename(mapper={'index':'filename'},axis=1)
    df['label'] = df['filename'].apply(label)
    return df[['filename', data_cat, 'label']]

def divide_signal(df, segment_length):
    '''
    This function divide the signal into segments, each with a specific number
    of points as defined by segment_length. Each segment will be added as an
    example (a row) in the returned DataFrame. Thus it increases the number of
    training examples. The remaining points which are less than segment_length
    are discarded.

    Parameter:
        df:
            DataFrame returned by matfile_to_df()
        segment_length:
            Number of points per segment.
    Return:
        DataFrame with segmented signals and their corresponding filename and
        label
    '''
    dic = {}
    idx = 0
    for i in range(df.shape[0]):
        n_sample_points = len(df.iloc[i,1])
        n_segments = n_sample_points // segment_length
        for segment in range(n_segments):
            dic[idx] = {
                'signal': df.iloc[i,1][segment_length * segment:segment_length * (segment+1)],
                'label': df.iloc[i,2],
                'filename' : df.iloc[i,0]
            }
            idx += 1
    df_tmp = pd.DataFrame.from_dict(dic,orient='index')
    df_output = pd.concat(
                [df_tmp[['label', 'filename']], 
                pd.DataFrame(np.vstack(df_tmp["signal"]))
                ], 
                axis=1 )
    return df_output


def normalize_signal(df, data_cat):
    '''
    Normalize the signals in the DataFrame returned by matfile_to_df() by subtracting
    the mean and dividing by the standard deviation.
    '''
    # DE_time
    mean = df[data_cat].apply(np.mean)
    std = df[data_cat].apply(np.std)
    df[data_cat] = (df[data_cat] - mean) / std

    return df

def get_df_all(data_path, data_cat, segment_length=512, normalize=False):
    '''
    Load, preprocess and return a DataFrame which contains all signals data and
    labels and is ready to be used for model training.

    Parameter:
        normal_path:
            Path of the folder which contains matlab files of normal bearings
        DE_path:
            Path of the folder which contains matlab files of DE faulty bearings
        segment_length:
            Number of points per segment. See divide_signal() function
        normalize:
            Boolean to perform normalization to the signal data
    Return:
        df_all:
            DataFrame which is ready to be used for model training.
    '''
    df = matfile_to_df(data_path, data_cat)

    if normalize:
        normalize_signal(df, data_cat)
    df_processed = divide_signal(df, segment_length)

    map_label = {'NORMAL':0, 'IR':1, 'OR':2, 'OR + IR':3}
    df_processed['label'] = df_processed['label'].map(map_label)
    return df_processed 
    
def download(url:str, dest_dir:Path, save_name:str, suffix=None) -> Path:
    assert isinstance(dest_dir, Path), "dest_dir must be a Path object"
    if not dest_dir.exists():
        dest_dir.mkdir()
    if save_name == None: filename = url.split('/')[-1]
    else: filename = save_name+suffix
    file_path = dest_dir / filename
    if not file_path.exists():
        print(f"Downloading {file_path}")
        with open(f'{file_path}', 'wb') as f:
            response = requests.get(url, stream=True)
            total = int(response.headers.get('content-length'))
            with tqdm(total=total, unit='B', unit_scale=True, desc=filename) as pbar:
                for data in response.iter_content(chunk_size=1024*1024):
                    f.write(data)
                    pbar.update(1024*1024)
    else:
        return file_path
    return file_path
