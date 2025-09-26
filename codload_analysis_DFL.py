# %% General Information

# Author: N. Bont

# Reviewed by: G.A. Oonk (29-04-2025)

# Sources: Databallpy (Github, https://pypi.org/project/databallpy/ & https://databallpy.readthedocs.io/en/latest/), ChatGPT

# Latest update: 26-09-2025



# %% Import packages

import pandas as pd

from databallpy import get_open_game

from databallpy.utils.constants import OPEN_GAME_IDS_DFL

from databallpy.features.differentiate import add_velocity, add_acceleration

from databallpy.features import get_covered_distance

from codload_functions import add_heading, create_summary_df, calculate_CODload

import numpy as np

import re

from pathlib import Path


# %% Load data and calculate all load metrics


def assign_team_name(player_id, game):

    if isinstance(player_id, str):

        if player_id.startswith("home_"):

            return game.home_team_name

        elif player_id.startswith("away_"):

            return game.away_team_name

    return "Unknown"


alle_match_metrics = []


# Loop: full analysis per match

for i, (game_id, game_name) in enumerate(OPEN_GAME_IDS_DFL.items(), start=1):

    try:

        # Load Open Source Bundesliga & 2. Bundesliga matches

        game = get_open_game(provider="metrica", game_id=game_id)


        # Save tracking ID's (tid)

        tracking_ids = game.get_column_ids()


        home_ids = [tid for tid in tracking_ids if tid.startswith("home_")]

        away_ids = [tid for tid in tracking_ids if tid.startswith("away_")]

        # Add ID and home/away status to players

        game.home_players["tracking_id"] = home_ids[: len(game.home_players)]

        game.away_players["tracking_id"] = away_ids[: len(game.away_players)]

        game.home_players["home_away"] = "home"

        game.away_players["home_away"] = "away"


        # Combine player home team and away team

        players = pd.concat([game.home_players, game.away_players], ignore_index=True)


        # Add team name

        players["team_name"] = players["tracking_id"].map(

            lambda pid: assign_team_name(pid, game)

        )


        # Filter / include active players

        active_ids = set(game.get_column_ids())

        players = players[players["tracking_id"].isin(active_ids)]


        # Determine column names

        column_names = game.tracking_data.columns

        player_columns = set(

            re.match(r"^(home_\d+|away_\d+)", col).group(1)
            
            for col in column_names

            if re.match(r"^(home_\d+|away_\d+)", col)

        )


        # Calculate velocity

        game.tracking_data = add_velocity(

            game.tracking_data,

            frame_rate=game.tracking_data.frame_rate,

            column_ids=player_columns,

            max_velocity=50.0,

            filter_type="moving_average",

            window_length=12,

        )


        # Calculate acceleration

        game.tracking_data = add_acceleration(

            game.tracking_data,

            frame_rate=game.tracking_data.frame_rate,

            column_ids=player_columns,

            max_acceleration=20.0,

            filter_type="moving_average",

            window_length=35,

            polyorder=2,

        )
        

        # Calculate heading

        game.tracking_data = add_heading(

            game.tracking_data,

            column_ids=game.get_column_ids(),

            frame_rate=game.tracking_data.frame_rate,

            filter_type="moving_average",

            window_length=7,

            polyorder=2,

            max_heading=360,

        )


        # Calculate covered distance per velocity zone

        Match_metrics = get_covered_distance(
            
            game.tracking_data,

            column_ids=game.get_column_ids(),

            frame_rate=game.tracking_data.frame_rate,

            velocity_intervals=((5.5, np.inf), (7, np.inf)),

        )


         # Calculate playing time in minutes and seconds

        x_columns = [col for col in game.tracking_data.columns if col.endswith('_x')]

        active_frames = game.tracking_data[x_columns].count()

        active_frames = active_frames[~active_frames.index.str.endswith("ball_x")]

        playing_time = pd.DataFrame({

             "PT_Seconds": active_frames / 25,

             "PT_Minutes": (active_frames / 25) / 60

        })

        playing_time.index = playing_time.index.str.replace("_x", "")

        Match_metrics = pd.concat([Match_metrics, playing_time], axis=1)
        
        
        # Calculate and add load metrics per minute

        Match_metrics['TD per min'] = Match_metrics['total_distance'] / Match_metrics['PT_Minutes']

        Match_metrics['RD per min'] = Match_metrics['total_distance_velocity_5.5_inf'] / Match_metrics['PT_Minutes']

        Match_metrics['SD per min'] = Match_metrics['total_distance_velocity_7_inf'] / Match_metrics['PT_Minutes']


        # Create summary_df needed for COD load calculation including heading change

        summary_df = create_summary_df(

            game.tracking_data,

            column_ids=game.get_column_ids(),

            frame_rate=game.tracking_data.frame_rate)
            
            
        # Adjust FrameRate to seconds

        game_periods_sec = game.periods.copy()

        frame_rate = game.tracking_data.frame_rate

        game_periods_sec['start_frame'] = (game_periods_sec['start_frame'] // frame_rate).astype(int)

        game_periods_sec['end_frame'] = (game_periods_sec['end_frame'] // frame_rate).astype(int)


        # Calculate COD Load

        CODload_df = calculate_CODload(

            summary_df,

            playing_time=playing_time,

            k=0.05,

            power=0.6,

            game_periods=game_periods_sec
        )

            
        CODload_df.set_index("PlayerID", inplace=True)


        # Combine load metrics with COD load in one df

        Match_metrics = pd.concat([Match_metrics, CODload_df], axis=1)

        Match_metrics = Match_metrics.reset_index().rename(columns={"index": "player_id"})


        # Add additional information per player

        Match_metrics = Match_metrics.merge(
            
            players[
                [
                    "tracking_id",
                   
                    "full_name",
                   
                    "position",
                   
                    "starter",
                   
                    "team_name",
                   
                    "home_away",
               
                ]
          
            ],
           
            left_on="player_id",
          
            right_on="tracking_id",
          
            how="left",
        
        )

        # Add Match Name

        Match_metrics["match_name"] = game_name

        Match_metrics = Match_metrics.drop(columns=[col for col in Match_metrics.columns if "Interval" in col])


        # Add calculated load metrics to df with load metric of all processed matches

        alle_match_metrics.append(Match_metrics)

        print(f"Loop {i} completed for match: {game.name}", flush=True)

    except Exception as e:

        print(f"Error in match: {game.name}: {e}")



# Combine all match data to a single df

final_df = pd.concat(alle_match_metrics, ignore_index=True)


# Save df to excel file

final_df.to_excel("LoadMetric_dataset.xlsx", index=False, engine="openpyxl")
