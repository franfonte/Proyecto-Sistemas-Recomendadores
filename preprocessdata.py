import pandas as pd
import os

def convert_all_dat_to_csv():
    """
    Converts all .dat files from the MovieLens 1M dataset 
    (ratings, movies, users) to CSV format.
    """
    base_path = os.path.join('data', 'ml-1m')
    
    # Check if the directory exists
    if not os.path.exists(base_path):
        print(f"Error: Directory not found at '{base_path}'")
        print("Please download and place the MovieLens 1M dataset in the 'data/' folder.")
        return

    # --- 1. Convert ratings.dat ---
    ratings_dat_path = os.path.join(base_path, 'ratings.dat')
    ratings_csv_path = os.path.join(base_path, 'ratings.csv')
    print(f"Converting '{ratings_dat_path}'...")
    
    ratings_df = pd.read_csv(
        ratings_dat_path, 
        sep='::', 
        header=None, 
        names=['userId', 'movieId', 'rating', 'timestamp'],
        engine='python',
        encoding='latin-1'
    )
    ratings_df.to_csv(ratings_csv_path, index=False)
    print(f"Successfully created '{ratings_csv_path}'")

    # --- 2. Convert movies.dat ---
    movies_dat_path = os.path.join(base_path, 'movies.dat')
    movies_csv_path = os.path.join(base_path, 'movies.csv')
    print(f"Converting '{movies_dat_path}'...")

    movies_df = pd.read_csv(
        movies_dat_path,
        sep='::',
        header=None,
        names=['movieId', 'title', 'genres'],
        engine='python',
        encoding='latin-1'
    )
    movies_df.to_csv(movies_csv_path, index=False)
    print(f"Successfully created '{movies_csv_path}'")

    # --- 3. Convert users.dat ---
    users_dat_path = os.path.join(base_path, 'users.dat')
    users_csv_path = os.path.join(base_path, 'users.csv')
    print(f"Converting '{users_dat_path}'...")

    users_df = pd.read_csv(
        users_dat_path,
        sep='::',
        header=None,
        names=['userId', 'gender', 'age', 'occupation', 'zipcode'],
        engine='python',
        encoding='latin-1'
    )
    users_df.to_csv(users_csv_path, index=False)
    print(f"Successfully created '{users_csv_path}'")
    
    print("\nAll conversions complete!")


if __name__ == "__main__":
    convert_all_dat_to_csv()