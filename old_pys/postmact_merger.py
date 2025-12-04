import os
import pickle
from collections import defaultdict
import glob
from tqdm import tqdm

# --- Configuration ---
# Point this to the directory where your chunk files are saved
data_directory = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/mact/" 
# The base prefix of your chunk files
file_prefix = "strat_dist_chunk" 
# The final, merged filename you want
final_output_prefix = "strat_dist_FINAL"
num_saes = 3
lyndex = [3, 6, 9]
sae_cp = [7000, 7000, 7000]

# --- Main Merging Logic ---
def merge_chunk_data():
    for i in range(num_saes):
        print(f"\n--- Merging data for SAE at Layer {lyndex[i]} ---")

        # Initialize master data structures for this SAE
        master_examples = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'examples': []}))
        master_counters = defaultdict(int)

        # Find all chunk files for the current SAE using a wildcard
        chunk_pattern = os.path.join(data_directory, f"{file_prefix}_{lyndex[i]}_{sae_cp[i]}_batch_*.pkl")
        chunk_files = glob.glob(chunk_pattern)
        
        if not chunk_files:
            print(f"No chunk files found for pattern: {chunk_pattern}")
            continue

        print(f"Found {len(chunk_files)} chunks to merge.")

        for chunk_file in tqdm(chunk_files, desc="Merging Chunks"):
            with open(chunk_file, "rb") as f:
                chunk_data = pickle.load(f)
            
            # Merge the total strata counters by adding them up
            for key, count in chunk_data['total_strata_counts'].items():
                master_counters[key] += count
            
            # Merge the word-specific examples and counts
            for key, word_dict in chunk_data['examples_and_word_counts'].items():
                for token_id, data in word_dict.items():
                    master_word_data = master_examples[key][token_id]
                    master_word_data['count'] += data['count']
                    # For simplicity, we'll just concatenate the example reservoirs.
                    # A more complex strategy could re-sample, but this is robust.
                    master_word_data['examples'].extend(data['examples'])

        # Save the final, merged data
        final_data_to_save = {
            'examples_and_word_counts': master_examples,
            'total_strata_counts': master_counters
        }
        
        final_save_path = os.path.join(data_directory, f"{final_output_prefix}_{lyndex[i]}_{sae_cp[i]}.pkl")
        with open(final_save_path, "wb") as f:
            pickle.dump(final_data_to_save, f)
        
        print(f"Successfully merged data. Final results saved to: {final_save_path}")

if __name__ == "__main__":
    merge_chunk_data()