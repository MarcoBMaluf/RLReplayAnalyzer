from carball.json_parser.game import Game
from carball.analysis.analysis_manager import AnalysisManager
import json 
import pandas as pd
import numpy as np
import subprocess
import os

# NEED TO FIND A WAY TO SET THESE IN THE BAKKESMOD APP
RRROCKET = '/Users/marcomaluf/Desktop/Unfinished Projects/New RL/rrrocket'
REPLAY_PATH = '/Users/marcomaluf/Desktop/Unfinished Projects/New RL/replays'
DATA_FOLDER_PATH = '/Users/marcomaluf/Desktop/Unfinished Projects/New RL/formatted data'

SHOT_SEGMENT = 64
REPLAYS = os.listdir(REPLAY_PATH)

def run_rrrocket(command):
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True
        )
        replay_json = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        # Print the error output
        print(f"Error: {e}")
        print(f"Return Code: {e.returncode}")
        print("Standard Error:")
        print(e.stderr)

    return replay_json


for replay in REPLAYS:
    try:
        print('---------------')
        print(f'Processing {replay}')

        command = [
            RRROCKET,
            "--network-parse",
            f'{REPLAY_PATH}/{replay}'
        ]

        replay_json = run_rrrocket(command)

        if 'PlayerStats' in replay_json['properties']:
            print('Formatting Data')
        else:
            print('Invalid Data Structure. Deleting file')
            os.remove(f'{REPLAY_PATH}/{replay}')
            continue

        game = Game()
        game.initialize(loaded_json=replay_json)

        analysis_manager = AnalysisManager(game)
        analysis_manager.create_analysis(calculate_intensive_events=False)
        replay_dataframe = analysis_manager.get_data_frame()

        highlights = replay_json['properties']['HighLights']
        keyframes = [h['frame'] for h in highlights]
        goals = replay_json['properties']['Goals']
        goalframes = [g['frame'] for g in goals]
        shots = {k: 0 for k in keyframes}
        for key in shots.keys():
            for g in goalframes:
                if key == g:
                    shots[key] = 1

        # DISTANCE FUNCTION FOR THE `dist_to_ball` COLUMN
        def distance(v1, v2):
            d = np.power(v1-v2, 2)
            d = np.sum(d, axis=1)

            dist = np.sqrt(d)
            
            return dist

        players_teams = {p['Name']: p['Team'] for p in replay_json['properties']['PlayerStats']}
        players_teams = dict(sorted(players_teams.items(), key=lambda item: item[1]))

        # COLUMNS I WANT (from ball and player)
        ball_cols = ['pos_x', 'pos_y', 'pos_z', 'vel_x','vel_y', 'vel_z', 
                    'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'hit_team_no']

        player_cols = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 
                    'ang_vel_x','ang_vel_y', 'ang_vel_z', 'throttle', 'steer', 
                    'dodge_active', 'double_jump_active', 'jump_active', 'boost_active']

        # merging the data together
        ball_data = replay_dataframe['ball'][ball_cols]
        match_data = replay_dataframe['ball'][ball_cols]
        for key, val in players_teams.items():
            temp_df = replay_dataframe[key][player_cols]
            temp_df['team'] = val
            temp_df['dist_to_ball'] = distance(ball_data[['pos_x', 'pos_y', 'pos_z']], 
                                            temp_df[['pos_x', 'pos_y', 'pos_z']])
            match_data = pd.concat([match_data, temp_df], axis=1)
        # resetting index
        match_data = match_data.reset_index(drop=True)

        # logic for if team size < 3
        if replay_json['properties']['TeamSize'] < 3:
            amt_to_add = 112 - len(match_data.columns)
            to_add = [f"Column_{i+1}" for i in range(amt_to_add)]
            match_data = pd.concat([match_data, pd.DataFrame(-9999999, index=range(match_data.shape[0]), columns=to_add)], axis=1)

        # adding label column
        match_data['label'] = np.nan
        # adding matchId for organizational purposes, will not be used in model
        matchId = replay_json['properties']['Id']
        match_data['matchId'] = matchId

        # getting the shot segments
        for key, val in shots.items():
            match_data.iloc[key, 112] = val

        for i, (k, v) in enumerate(shots.items()):
            frame = k-1
            if i == 0:
                formatted_data = match_data.iloc[frame-SHOT_SEGMENT:frame]
                formatted_data.label = v
            else:
                df = match_data.iloc[frame-SHOT_SEGMENT:frame]
                df.label = v
                formatted_data = pd.concat([formatted_data, df], axis=0)
            
        formatted_data.to_csv(f'{DATA_FOLDER_PATH}/{matchId}.csv')

    except Exception as e: 
        print(f'Error Occurred: {e}')