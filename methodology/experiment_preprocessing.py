import pandas as pd
import numpy as np
import os

'''
Required Functions:
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
    
def fill_matrix_with_zeros(matrix, num_players):
    
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

def keep_first_half(df, segment_size=64, keep=32):
    n, m = df.shape
    assert n % segment_size == 0
    
    arr = df.values.reshape(-1, segment_size, m)  # (K, 64, m)
    arr = arr[:, :keep, :]                         # (K, 32, m) — keep first 32 rows
    arr = arr.reshape(-1, m)                       # (K*32, m)
    
    return pd.DataFrame(arr, columns=df.columns)

# initializing data directories, list of CSV names
DATA_DIR = '/Users/marcomaluf/Desktop/Unfinished Projects/paper/training_replays_csvs'
SAVE_DIR = '/Users/marcomaluf/Desktop/Unfinished Projects/paper/training_preprocessed_data'
raw_csvs = os.listdir(DATA_DIR)
raw_csvs = [s for s in raw_csvs if '.csv' in s] # removing entries from the list of strings that aren't .csv files

for i, csv in enumerate(raw_csvs):
    print(f'--------CSV {i+1}--------')
    print(csv)
    sample_df = pd.read_csv(f'{DATA_DIR}/{csv}').drop(columns='Unnamed: 0')
    if sample_df.shape[0]%64 != 0:
        print(f'Skipping replay, not all segments are of equal length')
        continue

    '''
    Step 1: Get the ball positions, calculate speed, store in new principle dataframe
    '''
    speed = np.sqrt(np.sum([
    np.power(sample_df.vel_x.values.astype(float), 2),
    np.power(sample_df.vel_y.values.astype(float), 2),
    np.power(sample_df.vel_z.values.astype(float), 2)
    ], axis=0))

    ball_data = sample_df[['pos_x', 'pos_y', 'pos_z']].copy(deep=True)
    ball_data.loc[:,'speed'] = speed
    ball_data = ball_data.astype(float)

    '''
    Step 2: Using the new principle dataframe, for each time-step compute the 3D distance 
    of the ball to the center of the goal and the angle of the ball between the goal-posts
    '''
    final_value = []
    for i in range(64, ball_data.shape[0]+1, 64):
        final_value.append(ball_data['pos_y'].iloc[i-1])    
    half = np.where(np.array(final_value) < 0, 0, 1)
    blue_goal_center = np.array([0, -5200, 256])
    orange_goal_center = np.array([0, 5200, 256])

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
    Step 3: find the number of defending players in-between the ball and the goal 
    [in the 2D-triangle and with z <= 646, halfway between the floor and the ceiling]
    '''

    player_positions_indeces = [7,8,9,14, 16,17,18,23, 25,26,27,32, 34,35,36,41, 43,44,45,50, 52,53,54,59]
    player_segments = [sample_df.iloc[i:i+64, player_positions_indeces].copy(deep=True) for i in range(0, sample_df.shape[0], 64)]
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
    Step 4: number of players within 400 units of the ball
    '''
    col_indeces = [15, 24, 33, 42, 51, 60]
    dist_to_ball = sample_df.iloc[:, col_indeces].copy(deep=False)
    within_400 = []
    for i in range(6):
        within_400_indicator = np.select(condlist=[(dist_to_ball.iloc[:,i] <= 400) & (dist_to_ball.iloc[:,i] >= 0)], choicelist=[1], default=0)
        within_400.append(within_400_indicator)
    players_within_400 = np.sum(within_400, axis=0)
    ball_data.loc[:, 'players_within_400_of_ball'] = players_within_400

    '''
    Step 5: Spectral Embedding of the ball and the teams
    '''
    # getting ball and players positions
    num_players = find_player_count(sample_df)
    pos_cols = [0,1,2, 7,8,9, 16,17,18, 25,26,27, 34,35,36, 43,44,45, 52,53,54]
    ball_player_positions = sample_df.iloc[:, pos_cols[0:9+(3*(num_players-2))]].copy(deep=False)

    # computing similarity matrices for each time-step
    matrices = compute_distance_matrices(ball_player_positions, num_players)
    sim_matrices = RBF_kernel(matrices, sigma=0.0003)

    # setting similarties for players on different teams to 0
    for s in sim_matrices:
        s = fill_matrix_with_zeros(s, num_players)

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
    ball_data.loc[:,'team1_dist_to_ball_state'] = np.sqrt(np.power((ball_data.ball_state.values - ball_data.team1_cluster.values),2))
    ball_data.loc[:,'team2_dist_to_ball_state'] = np.sqrt(np.power((ball_data.ball_state.values - ball_data.team2_cluster.values),2))

    '''
    Step 6: Using the "real" units, make for both teams a column for each xyz centroid, 
    and add a column for each teams' centroids' distances to the ball
    '''
    player_pos_cols = [7,8,9, 16,17,18, 25,26,27, 34,35,36, 43,44,45, 52,53,54]

    def masked_mean(df, indices):
        subset = df.iloc[:, indices].copy().astype(float)
        
        # Track rows where ALL values are masked before replacing
        all_masked = (subset == -999999.00).all(axis=1)
        
        # Replace mask sentinel with NaN
        subset[subset == -999999.00] = np.nan
        
        # Fill NaN with row-wise nanmean of remaining valid columns
        row_means = subset.apply(lambda row: np.nanmean(row) if not np.all(np.isnan(row)) else np.nan, axis=1)
        for col in subset.columns:
            subset[col] = subset[col].fillna(pd.Series(row_means, index=subset.index))
        
        result = subset.mean(axis=1)
        
        # Restore masked sentinel where all values were originally masked
        result[all_masked] = -999999.00
        
        return result

    if num_players == 2:
        blue_x_index = [player_pos_cols[0]]
        blue_y_index = [player_pos_cols[1]]
        blue_z_index = [player_pos_cols[2]]

        orange_x_index = [player_pos_cols[3]]
        orange_y_index = [player_pos_cols[4]]
        orange_z_index = [player_pos_cols[5]]
    elif num_players == 4:
        blue_x_index = [player_pos_cols[i] for i in range(0,num_players+2,3)]
        blue_y_index = [player_pos_cols[i] for i in range(1,1+num_players+2,3)]
        blue_z_index = [player_pos_cols[i] for i in range(2,2+num_players+2,3)]

        orange_x_index = [player_pos_cols[i] for i in range(6,6+num_players+2,3)]
        orange_y_index = [player_pos_cols[i] for i in range(7,7+num_players+2,3)]
        orange_z_index = [player_pos_cols[i] for i in range(8,8+num_players+2,3)]
    else:
        blue_x_index = [player_pos_cols[i] for i in range(0,num_players+2,3)]
        blue_y_index = [player_pos_cols[i] for i in range(1,1+num_players+2,3)]
        blue_z_index = [player_pos_cols[i] for i in range(2,2+num_players+2,3)]

        orange_x_index = [player_pos_cols[i] for i in range(9,9+num_players+2,3)]
        orange_y_index = [player_pos_cols[i] for i in range(10,10+num_players+2,3)]
        orange_z_index = [player_pos_cols[i] for i in range(11,11+num_players+2,3)]

    blue_team_centroid = pd.concat([
        masked_mean(sample_df, blue_x_index[0:num_players//2]),
        masked_mean(sample_df, blue_y_index[0:num_players//2]),
        masked_mean(sample_df, blue_z_index[0:num_players//2])
    ], axis=1)

    orange_team_centroid = pd.concat([
        masked_mean(sample_df, orange_x_index[0:num_players//2]),
        masked_mean(sample_df, orange_y_index[0:num_players//2]),
        masked_mean(sample_df, orange_z_index[0:num_players//2])
    ], axis=1)

    blue_centroid_dist_to_ball = distance_3d(ball_data.iloc[:,0:3].values, blue_team_centroid.values)
    orange_centroid_dist_to_ball = distance_3d(ball_data.iloc[:,0:3].values, orange_team_centroid.values)

    ball_data.loc[:,'blue_x_centroid'] = blue_team_centroid.iloc[:,0]
    ball_data.loc[:,'blue_y_centroid'] = blue_team_centroid.iloc[:,1]
    ball_data.loc[:,'blue_z_centroid'] = blue_team_centroid.iloc[:,2]
    ball_data.loc[:,'blue_centroid_dist_to_ball'] = blue_centroid_dist_to_ball

    ball_data.loc[:,'orange_x_centroid'] = orange_team_centroid.iloc[:,0]
    ball_data.loc[:,'orange_y_centroid'] = orange_team_centroid.iloc[:,1]
    ball_data.loc[:,'orange_z_centroid'] = orange_team_centroid.iloc[:,2]
    ball_data.loc[:,'orange_centroid_dist_to_ball'] = orange_centroid_dist_to_ball

    ball_data

    '''
    Step 7: Add the label, and then trim the length of each sequence to be 32 frames
    '''
    ball_data.loc[:,'label'] = sample_df.label

    final_data = keep_first_half(ball_data)

    '''
    Step 8: Save the csvs for each game out into `training_preprocessed_data`
    '''
    final_data.to_csv(f'{SAVE_DIR}/{csv}.csv')

print('-------- Done --------')