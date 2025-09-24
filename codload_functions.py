# Folder with functions to calculate COD Load

from databallpy import get_game, get_open_game
from databallpy.features.differentiate import add_velocity, add_acceleration
from databallpy.features import get_individual_player_possession
from databallpy.features import get_covered_distance
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

def calculate_heading(
    tracking_data: pd.DataFrame,
    frame_rate: float,
    filter_type: str,
    window_length: int,
    polyorder: int,
    column_ids: list[str],
    max_heading: float,
    inplace: bool,
) -> pd.DataFrame:
    """Helper function to calculate heading based on position data"""
    for player_id in column_ids:
        # Calculate changes in x and y positions per second (not per frame)
        dx = (tracking_data[player_id + '_x'].diff()) / frame_rate  # Change in x per second
        dy = (tracking_data[player_id + '_y'].diff()) / frame_rate  # Change in y per second
        
        # Calculate heading in radians and convert to degrees
        heading = np.degrees(np.arctan2(dy, dx))  # Heading in degrees
        
        # Apply filtering if specified
        if filter_type == "moving_average":
            heading = pd.Series(heading).rolling(window=window_length, min_periods=1).mean()
        elif filter_type == "savitzky_golay":
            heading = savgol_filter(heading, window_length, polyorder)

        # Normalize heading to be within -180 to 180 degrees
        heading = (heading + 180) % 360 - 180  # Normalize to -180 to 180 degrees range

        # Clip heading values if they exceed the max_heading (if necessary)
        heading = np.clip(heading, -max_heading, max_heading)

        # Add heading to the tracking data
        tracking_data[player_id + '_heading'] = heading

    return tracking_data


def add_heading(
    tracking_data: pd.DataFrame,
    column_ids: str | list[str],
    frame_rate: float,
    filter_type: str = None,
    window_length: int = 7,
    polyorder: int = 2,
    max_heading: float = np.inf,
    inplace: bool = False,
) -> pd.DataFrame:
    """Function that adds heading columns based on the position columns

    Args:
        tracking_data (pd.DataFrame): tracking data of x- and y-coordinates
        column_ids (str | list[str]): columns for which heading should be calculated
        frame_rate (float): framerate of the tracking data
        filter_type (str, optional): filter type to use. Defaults to None.
            Options are `moving_average` and `savitzky_golay`.
        window_length (int, optional): window size for the filter. Defaults to 7.
        polyorder (int, optional): polynomial order for the filter. Defaults to 2.
        max_heading (float, optional): maximum value for the heading in degrees.
            Defaults to np.inf.
        inplace (bool, optional): whether to modify the DataFrame in place. Defaults

    Returns:
        pd.DataFrame: tracking data with the added heading columns if inplace is False
            else None.

    Raises:
        ValueError: if filter_type is not one of `moving_average`, `savitzky_golay`,
            or None.

    Note:
        The function will delete the columns in input_columns with the heading if
        they already exist.
    """

    if isinstance(column_ids, str):
        column_ids = [column_ids]

    if filter_type not in ["moving_average", "savitzky_golay", None]:
        raise ValueError(
            "filter_type should be one of: 'moving_average', "
            f"'savitzky_golay', None, got: {filter_type}"
        )

    # Calculate heading based on x and y positions (modified behavior)
    res_df = calculate_heading(
        tracking_data,
        frame_rate=frame_rate,
        filter_type=filter_type,
        window_length=window_length,
        polyorder=polyorder,
        column_ids=column_ids,
        max_heading=max_heading,
        inplace=inplace,
    )

    return res_df


def create_summary_df(tracking_data, column_ids, frame_rate):
    """
    Create a summary DataFrame with velocity (in m/s) and heading change per second for all players.

    Args:
        tracking_data (pd.DataFrame): Tracking data with position and heading columns.
        column_ids (list[str]): List of player column IDs (e.g., "home_14").
        frame_rate (int): Frame rate of the tracking data.

    Returns:
        pd.DataFrame: Summary DataFrame with velocity (m/s) and heading change (deg/s) per second.
    """
    tracking_data = tracking_data.copy()
    tracking_data.index = pd.date_range(start="2023-01-01", periods=len(tracking_data), freq=f"{1000 // frame_rate}ms")

    summary_data = {}

    for player_id in column_ids:
        heading_column = player_id + '_heading'
        velocity_column = player_id + '_velocity'

        resampled_data = tracking_data[[velocity_column, heading_column]].resample('1s').mean()

        # Bereken heading change met wrap-around correctie inline
        heading_diff = resampled_data[heading_column].diff()
        heading_diff = (heading_diff + 180) % 360 - 180  # Breng binnen [-180, 180]

        resampled_data['heading_change'] = heading_diff

        summary_data[player_id + '_vel'] = resampled_data[velocity_column]
        summary_data[player_id + '_hc'] = resampled_data['heading_change']

    summary_df = pd.DataFrame(summary_data)
    return summary_df

def calculate_CODload(summary_df, playing_time, k, power, game_periods):
    """
    Calculate COD load for all players, both overall COD load and per 15-minute interval.

    Args:
        summary_df (pd.DataFrame): DataFrame met velocity en heading change per seconde.
        playing_time (pd.DataFrame): DataFrame met PlayerID en PT_Seconds.
        k (float): Constante voor threshold velocity.
        power (float): Exponent voor heading change.
        game_periods (pd.DataFrame): DataFrame met kolommen start_frame, end_frame, start_datetime_td, end_datetime_td voor beide helften.
    
    Returns:
        pd.DataFrame: COD load per player for full match and per 15-minute interval.
    """

    # Gebruik game.periods in plaats van detect_halves
    first_half_end_idx = game_periods.loc[0, 'end_frame']
    second_half_start_idx = game_periods.loc[1, 'start_frame']
    first_half_end_time = game_periods.loc[0, 'end_datetime_td']
    second_half_start_time = game_periods.loc[1, 'start_datetime_td']

    # Define 15-minute intervals in frames
    fixed_intervals = [
        (0, 15 * 60),  # 0–15 min
        (15 * 60, 30 * 60),  # 15–30 min
        (30 * 60, first_half_end_idx),  # 30–45 min (tot einde eerste helft)
        (second_half_start_idx, second_half_start_idx + 15 * 60),  # 45–60 min
        (second_half_start_idx + 15 * 60, second_half_start_idx + 30 * 60),  # 60–75 min
        (second_half_start_idx + 30 * 60, len(summary_df))  # 75–90+ min
    ]

    interval_labels = ['COD load 0-15', 'COD load 15-30', 'COD load 30-45', 'COD load 45-60', 'COD load 60-75', 'COD load 75-90']
    results = []

    # Alle spelers identificeren
    player_ids = sorted(set('_'.join(col.split('_')[:2]) for col in summary_df.columns if '_vel' in col))

    for player_id in player_ids:
        player_result = {"PlayerID": player_id}
        vel_col = f"{player_id}_vel"
        head_col = f"{player_id}_hc"

        if vel_col not in summary_df.columns or head_col not in summary_df.columns:
            print(f"Warning: Missing columns for {player_id}")
            continue

        velocity = summary_df[vel_col]
        heading = summary_df[head_col].abs().clip(lower=1e-5)

        # COD load hele wedstrijd
        threshold = 1 / (k * (heading ** power))
        dis_to_threshold = velocity - threshold
        cod_sum = dis_to_threshold[dis_to_threshold > 0].sum()

        try:
            pt = playing_time.loc[player_id, "PT_Seconds"]
        except KeyError:
            pt = None
            print(f"Warning: No playtime for {player_id}")

        player_result["COD load"] = cod_sum / pt if pt and pt > 0 else 0

        # COD load per 15-minuten interval
        for i, (start, end) in enumerate(fixed_intervals):
            interval_vel = velocity.iloc[start:end]
            interval_head = heading.iloc[start:end]

            if interval_vel.notna().sum() == 0 or interval_head.notna().sum() == 0:
                cod_load = np.nan
            else:
                threshold_i = 1 / (k * (interval_head ** power))
                dis_to_threshold_i = interval_vel - threshold_i
                cod_sum_i = dis_to_threshold_i[dis_to_threshold_i > 0].sum()
                interval_secs = interval_vel.notna().sum()
                cod_load = cod_sum_i / interval_secs if interval_secs > 0 else 0

            label = interval_labels[i]
            player_result[label] = cod_load
            player_result[f"Interval_{label}"] = (
                f"{str(summary_df.index[start])} - {str(summary_df.index[min(end, len(summary_df)-1)])}"
            )

        results.append(player_result)

    return pd.DataFrame(results)