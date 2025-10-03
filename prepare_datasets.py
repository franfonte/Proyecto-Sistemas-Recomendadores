import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split as surprise_train_test_split

# --- Configuración Global ---
BASE_DATA_PATH = os.path.join('data', 'ml-1m')
ORIGINAL_RATINGS_FILE = os.path.join(BASE_DATA_PATH, 'ratings.csv')
RANDOM_SEED = 42

def create_dataset_subsets():
    """
    Reads the original ratings.csv and creates 5 directories with subsets
    of the data (10%, 25%, 50%, 75%, 100%).
    Uses stratified sampling to maintain user rating distribution.
    """
    if not os.path.exists(ORIGINAL_RATINGS_FILE):
        print(f"Error: Original ratings file not found at '{ORIGINAL_RATINGS_FILE}'")
        print("Please run 'preprocess_data.py' first.")
        sys.exit(1)

    print("--- Starting dataset subset creation ---")
    df = pd.read_csv(ORIGINAL_RATINGS_FILE)
    
    percentages = [10, 25, 50, 75, 100]

    for p in percentages:
        subset_dir = os.path.join('data', str(p))
        os.makedirs(subset_dir, exist_ok=True)
        output_path = os.path.join(subset_dir, 'ratings.csv')

        if p == 100:
            # For 100%, just copy the original file
            subset_df = df
        else:
            # Stratified sampling based on userId
            # We use sklearn_train_test_split as a convenient way to do stratified sampling
            subset_df, _ = sklearn_train_test_split(
                df,
                train_size=(p / 100.0),
                stratify=df['userId'],
                random_state=RANDOM_SEED
            )
        
        subset_df.to_csv(output_path, index=False)
        print(f"Successfully created '{output_path}' with {len(subset_df)} ratings ({p}%)")
    
    print("--- All subsets created successfully ---\n")


def split_and_generate_antitest(subset_dir_path):
    """
    Takes a directory with a ratings.csv file, splits it into 80/20
    train/test sets, and generates an anti-test set from the train set.
    Saves train.csv, test.csv, and antitest.csv in the same directory.
    """
    ratings_file = os.path.join(subset_dir_path, 'ratings.csv')
    print(f"--- Processing directory: {subset_dir_path} ---")

    # 1. Load data with Surprise
    df = pd.read_csv(ratings_file)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    
    # 2. Split into train and test sets (80/20)
    trainset, testset = surprise_train_test_split(data, test_size=0.20, random_state=RANDOM_SEED)

    # 3. Generate anti-test set (pairs not in the trainset)
    anti_testset = trainset.build_anti_testset()

    # 4. Convert back to DataFrames and save
    # Train set
    train_df = pd.DataFrame(trainset.all_ratings(), columns=['userId_inner', 'movieId_inner', 'rating'])
    train_df['userId'] = train_df['userId_inner'].apply(trainset.to_raw_uid)
    train_df['movieId'] = train_df['movieId_inner'].apply(trainset.to_raw_iid)
    train_df[['userId', 'movieId', 'rating']].to_csv(os.path.join(subset_dir_path, 'train.csv'), index=False)
    
    # Test set
    test_df = pd.DataFrame(testset, columns=['userId', 'movieId', 'rating'])
    test_df.to_csv(os.path.join(subset_dir_path, 'test.csv'), index=False)

    # Anti-test set
    anti_test_df = pd.DataFrame(anti_testset, columns=['userId', 'movieId', 'rating_placeholder'])
    # Add empty rating column and original timestamp for format consistency
    anti_test_df['rating'] = '' 
    anti_test_df = anti_test_df[['userId', 'movieId', 'rating']] # Reorder to match original
    anti_test_df.to_csv(os.path.join(subset_dir_path, 'antitest.csv'), index=False)

    print(f"  -> Created train.csv, test.csv, and antitest.csv")

if __name__ == "__main__":
    # Step 1: Create the 5 directories with data subsets
    create_dataset_subsets()

    # Step 2: For each subset, split into train/test and create antitest
    percentages = [10, 25, 50, 75, 100]
    for p in percentages:
        subset_dir = os.path.join('data', str(p))
        split_and_generate_antitest(subset_dir)
        
    print("\n--- All datasets have been prepared successfully! ---")