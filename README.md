# Change of Direction Load in Football Players: a Validation Study in 293 Elite-Level Matches
This is the official repository for the paper:
> Bont, N., Den Otter, A. R., Oonk, G. A., & Brink, M. S. (2025). Change of Direction Load in Football Players: a Validation Study in 293 Elite-Level Matches. _Not Sumbitted._

## Definition: Change of Direction (COD) load
Change of direction load is a recently developed load indicator combining heading change and velocity (Merks et al., 2022). 

## Project Structure
This project includes two Python codes.
 
- `codload_functions.py`: Required functions to analyze a dataset with position tracking data into COD loads.

**def calculate_heading** : Helper function to calculate heading based on position data.

**def add_heading** : Function that adds heading columns based on the position columns.

**def create_summary_df** : Function to create a DataFrame with velocity (in m/s) and heading change per second for all players.

**def calculate_CODload** : Function to calculate COD load for all players, both overall COD load and per 15-minute interval.


- `codload_analysis_DFL.py`: Code which presents the full analysis applied on an open-source dataset of 7 matches in the German Bundesliga (Bassek et al., 2025). This code is derived from the code used for analysis in our paper.

## Licence
This work is licensed under a Creative Commons Attribution 4.0 International License ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
You must therefore give appropriate credit when using this dataset by citing this publication.

## References
> Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. Sci Data, 12(1), 195. https://doi.org/10.1038/s41597-025-04505-y 

> Merks, B. M. T., Frencken, W. G. P., Den Otter, A. R., & Brink, M. S. (2022). Quantifying change of direction load using positional data from small-sided games in soccer. Science & medicine in football, 6(2), 234-240. https://doi.org/doi:10.1080/24733938.2021.1912382 
