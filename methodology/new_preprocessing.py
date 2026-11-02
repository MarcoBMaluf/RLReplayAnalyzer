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

def sign(p1, p2, p3):
    """Computes the cross product sign for all timesteps."""
    result = (p1[:, 0] - p3[:, 0]) * (p2[:, 1] - p3[:, 1]) - (p2[:, 0] - p3[:, 0]) * (p1[:, 1] - p3[:, 1])
    return result

def is_player_inside_ball_goal_triangle(player_pos, ball_position, half):

    # Define goalpost positions based on which half
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
    p_right = np.tile(p_right, (player_pos.shape[0], 1))  # (N, 3)
    
    # Compute signs for the three edges of the triangle
    s1 = sign(player_pos, ball_position, p_left)
    s2 = sign(player_pos, p_left, p_right)
    s3 = sign(player_pos, p_right, ball_position)
    
    inside = ((s1>=0) & (s2>=0) & (s3>=0)) | ((s1<=0) & (s2<=0) & (s3<=0))
    return inside

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
    
    if df.iloc[0, 25] == -9999999:
        return 2
    elif df.iloc[0, 43] == -9999999:
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
DATA_DIR = '/Users/marcomaluf/Desktop/Unfinished Projects/paper/formatted_data'
FORMATTED_DIR = '/Users/marcomaluf/Desktop/Unfinished Projects/paper/preprocessed_data'
raw_csvs = os.listdir(DATA_DIR)
raw_csvs = [s for s in raw_csvs if '.csv' in s] # removing entries from the list of strings that aren't .csv files

# iterating through the CSVs
for i, csv in enumerate(raw_csvs):
    print(f'--------CSV {i+1}--------')

    '''
    Step 0:
    (Load in match dataframe and only select shots/goals (i.e. where label != -1))
    '''
    df = pd.read_csv(f'{DATA_DIR}/{csv}').drop(columns='Unnamed: 0')
    # df = pd.read_csv('/Users/marcomaluf/Desktop/Unfinished Projects/paper/formatted_data/APAC P-F SE vs Star G1 2026-01-18.14.41.csv').drop(columns='Unnamed: 0')

    '''
    Step 1:
    (Separate relevant ball data into its own dataframe (will be the principle dataframe post-separation), 
    and reset the index)
    '''
    ball_data = df[['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']].copy(deep=True)
    ball_data.reset_index(inplace=True, drop=True)

    '''
    Step 2:
    (Using the new principle dataframe, for each time-step compute the 3D distance of the ball 
    to the center of the goal and the angle of the ball between the goal-posts)
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
    (Using the new angle variable, use the player positions of the defending team 
    to determine (at each time-step) how many of them are betweeen the ball and the goal)
    '''
    # pos_x, pos_y, pos_z, team...
    player_positions_indeces = [7,8,9,14, 16,17,18,23, 25,26,27,32, 34,35,36,41, 43,44,45,50, 52,53,54,59]
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
    (Find the number of players within 400 units of the ball at each time-step)
    '''
    col_indeces = [15, 24, 33, 42, 51, 60]
    dist_to_ball = df.iloc[:, col_indeces].copy(deep=False)

    within_400 = []
    for i in range(6):
        within_400_indicator = np.select(condlist=[dist_to_ball.iloc[:,i] <= 400], choicelist=[1], default=0)
        within_400.append(within_400_indicator)

    players_within_400 = np.sum(within_400, axis=0)
    ball_data.loc[:, 'players_within_400_of_ball'] = players_within_400

    '''
    Step 5:
    (Implement a Spectral Embedding on the attacking and defending teams, respectively, to get a team-state variable)
    '''
    # getting ball and players positions
    num_players = find_player_count(df)
    pos_cols = [0,1,2, 7,8,9, 16,17,18, 25,26,27, 34,35,36, 43,44,45, 52,53,54]
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

    """
    1/29/26 NOTE: 
    - reset the index of ball_data, then add the state columns. After, add a final column for the base_df index values (df.index, 
    df being the input df from step 0)

    2/2/26 CHANGE: 
    - no need to reset index, but will add column at the end
    """

    # adding columns
    ball_data.loc[:,'ball_state'] = ball_state
    ball_data.loc[:,'team1_cluster'] = team1_cluster
    ball_data.loc[:,'team2_cluster'] = team2_cluster

    '''
    Step 6:
    (Once all the new features are created, reformat the ball data (absolute value of the `y` direction), 
    fill in NA values, and standardize/scale the features appropriately for use with a model. 
    Add the label into the final dataset, and add the base_df indeces as the final column.)
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
    ball_data.loc[:, 'label'] = df.label.reset_index(drop=True)
    ball_data.loc[:, 'base_df_index'] = df.base_df_index.reset_index(drop=True)

    '''
    Step 7
    (Save the new dataframe to the `preprocessed_data` folder.)
    '''

    ball_data.to_csv(f'{FORMATTED_DIR}/{csv}.csv')

print('-------- Done --------')

