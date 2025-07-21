# import packages
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
import torch
from torch.utils.data import TensorDataset
    
'''
functions used in the preprocessing algo:
'''
def distance_3d(v1, v2):
    d = np.power(v1-v2, 2)
    d = np.sum(d, axis=1)

    distance = np.sqrt(d)
    
    return distance

def ball_goal_angle(a, b = np.array([-1000, 5200]), c = np.array([1000, 5200])):
    # a := ball 2d position, b := left post, c := right post
    a = a.copy()  
    a['pos_y'] = np.abs(a['pos_y'])
    a = a[['pos_x', 'pos_y']].values  

    ab = b - a
    ac = c - a
    dot = np.einsum('ij,ij->i', ab, ac) 

    mag_ab = np.linalg.norm(ab, axis=1)
    mag_ac = np.linalg.norm(ac, axis=1)

    theta = np.degrees(np.arccos(dot / (mag_ab * mag_ac)))

    return theta  

def is_player_inside_ball_goal_triangle(player_pos, ball_position, half):

    def sign(p1, p2, p3):
        """Computes the cross product sign for all timesteps."""
        return (p1[:, 0] - p3[:, 0]) * (p2[:, 1] - p3[:, 1]) - (p2[:, 0] - p3[:, 0]) * (p1[:, 1] - p3[:, 1])
    
    if half == 0:
        p_left = np.array([-1000, -5200, 646])  # Left goalpost
        p_right = np.array([1000, -5200, 646])  # Right goalpost
    else:
        p_left = np.array([-1000, 5200, 646])
        p_right = np.array([1000, 5200, 646])

    # Ensure inputs are NumPy arrays
    player_pos = np.asarray(player_pos)  # Shape (N, 3)
    ball_position = np.asarray(ball_position)  # Shape (N, 3)

    # Expand goalpost positions to match shape (N, 3)
    p_left = np.tile(p_left, (player_pos.shape[0], 1))   # (N, 3)
    p_right = np.tile(p_right, (player_pos.shape[0], 1)) # (N, 3)

    # Compute the cross-product signs for each timestep
    s1 = sign(player_pos, p_left, p_right)
    s2 = sign(player_pos, p_right, ball_position)
    s3 = sign(player_pos, ball_position, p_left)

    # Player is inside if all signs are the same (either all positive or all negative)
    return (s1 >= 0) & (s2 >= 0) & (s3 >= 0) | (s1 <= 0) & (s2 <= 0) & (s3 <= 0)

def RBF_kernel(distance, sigma):

    s = np.exp(-1 * (distance**2/2*sigma**2))

    return s

def compute_distance_matrices(df, num_players):
    num_entities = num_players + 1  # 6 players + 1 ball

    # Reshape the dataframe into (timesteps, entities, 3)
    positions = df.values.reshape(len(df), num_entities, 3)  # (T, 7, 3)

    # Compute pairwise distances: Euclidean norm along axis=2 after broadcasting
    dist_matrices = np.linalg.norm(positions[:, :, np.newaxis, :] - positions[:, np.newaxis, :, :], axis=-1)  # (T, 7, 7)

    return dist_matrices

def find_player_count(df):
    
    if df.iloc[0, 44] == -9999999:
        return 2
    elif df.iloc[0, 78] == -9999999:
        return 4
    else:
        return 6
    
def fill_matrix_with_zeros(matrix):
    
    for i in range(1, matrix.shape[0]):
        if i < matrix.shape[0]/2:
            matrix[i][int(num_players/2+1):num_players+1] = 0
        if i > matrix.shape[0]/2:
            matrix[i][1:int(num_players/2+1)] = 0

    return matrix

def spectral_embedding_1d(L):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Get the second smallest eigenvector (first non-trivial one)
    X = eigenvectors[:, np.argsort(eigenvalues)[1]]  # 1D vector

    return X

"""
Running the Preprocessing Algo:
"""

# getting data directories, list of CSV names
DATA_DIR = '/Users/marcomaluf/Desktop/Unfinished Projects/New RL/raw CSVs'
FORMATTED_DIR = '/Users/marcomaluf/Desktop/Unfinished Projects/New RL/datasets/LSTM1_preprocessing'
raw_csvs = os.listdir(DATA_DIR)
raw_csvs = [s for s in raw_csvs if '.csv' in s] # removing entries from the list of strings that aren't .csv files

# initializing lists for torch dataset:
torch_set_features = []
torch_set_labels = []

# iterating through the CSVs
for i, csv in enumerate(raw_csvs):
    print(f'--------CSV {i+1}--------')

    df = pd.read_csv(f'{DATA_DIR}/{csv}').drop(columns='Unnamed: 0')
    df = df.fillna(0)

    '''
    Step 1:
    '''
    ball_data = df[['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']].copy(deep=True)

    '''
    Step 2:
    '''
    final_value = []
    for i in range(64, ball_data.shape[0]+1, 64):
        final_value.append(ball_data['pos_y'].iloc[i-1])
        
    half = np.where(np.array(final_value) < 0, 0, 1)
    blue_goal_center = np.array([0, -5200, 273])
    orange_goal_center = np.array([0, 5200, 273])

    segments = [ball_data[['pos_x', 'pos_y', 'pos_z']].iloc[i:i+64] for i in range(0, ball_data.shape[0], 64)]
    for seg, h in zip(segments, half):
        if h == 0:
            d = distance_3d(seg, blue_goal_center)
            seg['distance_to_goal'] = d
        else:
            d = distance_3d(seg, orange_goal_center)
            seg['distance_to_goal'] = d
    dist_to_goal = pd.concat(segments, ignore_index=True)
    dist_to_goal = dist_to_goal['distance_to_goal']
    ball_data.loc[:, 'dist_to_goal'] = dist_to_goal.values

    angle_to_goal = ball_goal_angle(a=ball_data[['pos_x', 'pos_y']])
    ball_data.loc[:, 'angle_to_goal'] = angle_to_goal

    '''
    Step 3:
    '''
    player_positions_indeces = [10,11,12,25, 27,28,29,42, 44,45,46,59, 61,62,63,76, 78,79,80,93, 95,96,97,110]
    player_segments = [df.iloc[i:i+64, player_positions_indeces].copy(deep=True) for i in range(0, df.shape[0], 64)]

    players_btwn_ball_and_goal = []
    for ball, players, h in zip(segments, player_segments, half):
        temp = []
        
        # Identify player indices (assuming 4 columns per player: x, y, z, team)
        num_players = players.shape[1] // 4  
        selected_columns = []  

        for i in range(num_players):
            team_col = i * 4 + 3  # The "team" column index for player i
            
            # Keep players whose team is NOT on the ball's half
            if not (players.iloc[:, team_col] == h).all():
                selected_columns.extend(players.columns[i * 4: i * 4 + 4])

        # Filter the DataFrame
        players = players[selected_columns]

        # Iterate over each player and check if they are inside the ball-goal triangle
        for i in range(0, players.shape[1] - 3, 4):  # Step by 4 to get x, y, z
            player = players.iloc[:, i:i+3]  # Select x, y, z only
            temp.append(is_player_inside_ball_goal_triangle(player, ball, h))

        temp = np.sum(temp, axis=0)
        players_btwn_ball_and_goal.append(temp)

    ball_data.loc[:, 'players_btwn_ball_and_goal'] = np.hstack(players_btwn_ball_and_goal)

    '''
    Step 4:
    '''
    col_indeces = [26, 43, 60, 77, 94, 111]
    dist_to_ball = df.iloc[:, col_indeces].copy(deep=False)

    within_400 = []
    for i in range(6):
        within_400_indicator = np.select(condlist=[dist_to_ball.iloc[:,i] <= 400], choicelist=[1], default=0)
        within_400.append(within_400_indicator)

    players_within_400 = np.sum(within_400, axis=0)
    ball_data.loc[:, 'players_within_400_of_ball'] = players_within_400

    '''
    Step 5:
    '''
    # getting ball and players positions
    num_players = find_player_count(df)
    pos_cols = [0,1,2, 10,11,12, 27,28,29, 44,45,46, 61,62,63, 78,79,80, 95,96,97]
    ball_player_positions = df.iloc[:, pos_cols[0:9+(3*(num_players-2))]].copy(deep=False)

    # computing similarity matrices for each time-step
    matrices = compute_distance_matrices(ball_player_positions, num_players)
    sim_matrices = RBF_kernel(matrices, sigma=0.0003)

    # setting similarties for players on different teams to 0
    for s in sim_matrices:
        s = fill_matrix_with_zeros(s)

    # computing laplacian matrices
    laplacians = []
    for s in sim_matrices:
        d = np.diag(s.sum(axis=1))
        l = d - s
        laplacians.append(l)

    # finding the bottom 1 eigenvector for each time-step (i.e. for each matrix)
    eigvecs = []
    for l in laplacians:
        v = spectral_embedding_1d(l)
        eigvecs.append(v)
            
    # storing the vectors in a dataframe object to concat to the main dataframe (each value in the vector is in a separate column)
    eigvec_df = pd.DataFrame(eigvecs)

    # separating ball_state and each team's clusters 
    ball_state = eigvec_df.iloc[:,0].copy(deep=False)
    team1_cluster = np.sum(eigvec_df.iloc[:,1:int(num_players/2+1)], axis=1)/(num_players/2)
    team2_cluster = np.sum(eigvec_df.iloc[:,int(num_players/2+1):num_players+1], axis=1)/(num_players/2)

    # adding columns
    ball_data.loc[:,'ball_state'] = ball_state
    ball_data.loc[:,'team1_cluster'] = team1_cluster
    ball_data.loc[:,'team2_cluster'] = team2_cluster

    '''
    Step 6:
    '''
    standardize_cols = ['pos_x','pos_y','pos_z','vel_x','vel_y','vel_z']
    scale_cols = ['dist_to_goal','angle_to_goal','players_btwn_ball_and_goal','players_within_400_of_ball']
    nothing = ['ball_state','team1_cluster','team2_cluster']

    preprocessor = ColumnTransformer([
        ('standard', StandardScaler(), standardize_cols),
        ('minmax', MinMaxScaler(), scale_cols),
    ], remainder='passthrough')

    transformed_values = preprocessor.fit_transform(ball_data)

    ball_data[standardize_cols + scale_cols] = transformed_values[:, :len(standardize_cols) + len(scale_cols)]
    ball_data.loc[:, 'label'] = df.label

    '''
    Step 7
    '''
    SEQ_LENGTH = 64
    NUM_SEQS = int(ball_data.shape[0]/64)

    for i in range(NUM_SEQS):
        data = ball_data.iloc[int(64*i):int(64+64*i),0:13].copy(deep=False)
        label = ball_data.iloc[int(64*i):int(64+64*i)].label.values[0]

        torch_set_features.append(data)
        torch_set_labels.append(label)
    
    print('-------- Done --------')

# Convert DataFrames to tensors
sequences = torch.stack([torch.tensor(df.values, dtype=torch.float32) for df in torch_set_features])
labels_tensor = torch.tensor(torch_set_labels, dtype=torch.float32)  # Use float32 for regression or long for classification

# Use TensorDataset to create a dataset
dataset = TensorDataset(sequences, labels_tensor)

torch.save(dataset, f'{FORMATTED_DIR}/dataset.pt')