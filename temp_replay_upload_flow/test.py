from carball.json_parser.game import Game
from carball.analysis.analysis_manager import AnalysisManager
import subprocess
import json
import pandas as pd
import numpy as np

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


RRROCKET = '/Users/marcomaluf/Desktop/Unfinished Projects/New RL/rrrocket'
REPLAY = '/Users/marcomaluf/Desktop/Unfinished Projects/New RL/temp_replay_upload_flow/63fff9b2-c623-4809-b213-6026150bb7c5.replay copy'

command = [
    RRROCKET,
    "--network-parse",
    f'{REPLAY}'
]

replay_json = run_rrrocket(command)

game = Game()
game.initialize(loaded_json=replay_json)

analysis_manager = AnalysisManager(game)
analysis_manager.create_analysis(calculate_intensive_events=False)
replay_dataframe = analysis_manager.get_data_frame()

replay_dataframe.info(verbose=True)
# replay_dataframe.to_csv('/Users/marcomaluf/Desktop/Unfinished Projects/New RL/temp_replay_upload_flow/test_df2.csv')

# players_teams = {p['Name']: p['Team'] for p in replay_json['properties']['PlayerStats']}
# players_teams = dict(sorted(players_teams.items(), key=lambda item: item[1]))

# # COLUMNS I WANT (from ball and player)
# ball_cols = ['pos_x', 'pos_y', 'pos_z', 'vel_x','vel_y', 'vel_z', 
#             'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'hit_team_no']

# player_cols = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 
#             'ang_vel_x','ang_vel_y', 'ang_vel_z', 'throttle', 'steer', 
#             'dodge_active', 'double_jump_active', 'jump_active', 'boost_active']

# # merging the data together
# ball_data = replay_dataframe['ball'][ball_cols]
# match_data = replay_dataframe['ball'][ball_cols]
# for key, val in players_teams.items():
#     temp_df = replay_dataframe[key][player_cols]
#     temp_df['team'] = val
#     temp_df['dist_to_ball'] = distance(ball_data[['pos_x', 'pos_y', 'pos_z']], 
#                                     temp_df[['pos_x', 'pos_y', 'pos_z']])
#     match_data = pd.concat([match_data, temp_df], axis=1)
# # resetting index
# match_data = match_data.reset_index(drop=True)