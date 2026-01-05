import pickle
import os
from collections import Counter
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

#region Configuration

# Hardware settings
MAX_WORKERS = 18  # Leaves 2 cores free for OS/navigation
FEAT_RP = "/home/strah/Documents/Work_related/thon-of-py/blober/features/"

# The exact list of words you want to find
GEN_CHECK_LIST = [
    "male","female", "man","woman", "boy","girl", "mr","mrs", "ms","miss",
    "sir","madam", "gentleman","lady", "husband","wife", "father","mother",
    "son","daughter", "brother","sister", "grandfather","grandmother",
    "grandson","granddaughter", "widower","widow", "fiancé","fiancée",
    "patriarch","matriarch", "landlord","landlady", "he","she", "him","her",
    "his","hers", "himself","herself",
]

# The pre-computed Token IDs corresponding to the list above
ID_LIST = [
    3287, 2931, 2158, 2450, 2879, 2611, 2720, 3680, 5796, 3335, 
    2909, 21658, 10170, 3203, 3129, 2564, 2269, 2388, 2365, 2684, 
    2567, 2905, 5615, 7133, 7631, 12787, 7794, 7794, 19154, 19455, 
    12626, 13523, 18196, 2455, 2002, 2016, 2032, 2014, 2010, 5106, 
    2370, 2841
]

# Create a fast lookup set for workers (O(1) lookup time)
TARGET_ID_SET = set(ID_LIST)

# Create a mapping for the final results (ID -> Word)
ID_TO_WORD_MAP = dict(zip(ID_LIST, GEN_CHECK_LIST))

#endregion

#region Main Funcs

def process_single_file(file_path):
    """
    Optimized worker:
    1. Loads file (Single Feature)
    2. Performs Integer lookup (No decoding!)
    3. Returns only matches
    """
    results = [] # Stores: (feature_idx, strata_idx, token_id)
    
    try:
        # Extract feature ID from filename so we know where this came from
        # e.g., ".../feature_123.pkl" -> 123
        try:
            base_name = os.path.basename(file_path)
            # robustly get the number between '_' and '.'
            feat_idx = int(base_name.split('_')[-1].split('.')[0])
        except:
            feat_idx = -1 # Fallback if naming convention fails

        with open(file_path, "rb") as f:
            feature = pickle.load(f)
            

        #Doing single feat
        if not feature: 
            return ("SUCCESS", [])

        #! Examples are a DICTIONARY
        examples = feature.get("examples", {})
        
        # We iterate explicitly over keys 0, 1, 2. 
        # This prevents the 'int is not iterable' error caused by enumerate() on a dict keys.
        for strata_idx in [0, 1, 2]:
            strata = examples.get(strata_idx)

            if not strata: continue
            
            for entry in strata:
                # INTEGER CHECK (Fastest possible check)
                tid = entry.get("token_id")
                val = entry.get("value")
                
                if tid in TARGET_ID_SET:
                    # Found a match! Record location and ID.
                    results.append( (feat_idx, strata_idx, val ,tid) )

    except Exception as e:
        return ("ERROR", file_path, str(e))

    return ("SUCCESS", results)


def main():


    #region* Generate File List
    

    span_counts = 4
    search_span = [(6144//span_counts)*i for i in range(1,span_counts+1)]

    print("Generating file list...")

    # Find existing file paths
    path_ls = []
    for i in range(len(search_span)):
        if(search_span[i] == search_span[0]):
            s_point = 0
        else:
            s_point = search_span[i-1]
        e_point = search_span[i]

        for j in range(s_point, e_point):
            lp = FEAT_RP + f"feature_{j}.pkl"
            # Only add existing files to avoid errors in workers
            if os.path.exists(lp):
                path_ls.append(lp)

    print(f"Found {len(path_ls)} files to process.")
    print(f"Scanning for {len(TARGET_ID_SET)} unique Token IDs across {MAX_WORKERS} cores.")


    #endregion* Generate File List



    #region* Execution (Process Pool)
    

    cands = [[], [], []] # Defining per strata
    
    # Use 'spawn' context if on Linux to avoid fork issues with huge memory, 
    # though standard 'fork' (default on Linux) is usually faster for simple tasks.
    # If it hangs, try changing to get_context('spawn').
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results_iter = list(tqdm(
            executor.map(process_single_file, path_ls), 
            total=len(path_ls),
            unit="file",
            desc="Processing Files"
        ))


    #endregion* Execution (Process Pool)



    #region* Aggregation (The "Reduce" Step)
    

    print("Aggregating results...")
    
    errors = 0
    total_found = 0
    
    for item in results_iter:
        status = item[0]


        if status == "ERROR":
            errors += 1
            continue
        
        
        payload = item[1]

        # payload is list of (feat_idx, strata_idx, tid)
        for (feat_idx, strata_idx, val, tid) in payload:
            if strata_idx < 3:
                # Retrieve the word string from our pre-computed dictionary
                word_str = ID_TO_WORD_MAP.get(tid, "UNKNOWN")
                
                # Store exactly as your original code did: (token_id, string)
                # Note: We aren't storing feat_idx here to match your original output structure,
                # but we have it available if you change your mind!
                cands[strata_idx].append((feat_idx,val, tid, word_str))
                total_found += 1


    #endregion* Aggregation (The "Reduce" Step)



    #region* Fin Report
    
    print("-" * 30)
    print("DONE")
    print(f"Total Errors: {errors}")
    print(f"Strata 0 Candidates: {len(cands[0])}")
    print(f"Strata 1 Candidates: {len(cands[1])}")
    print(f"Strata 2 Candidates: {len(cands[2])}")
    print(f"Total Found: {total_found}")
    print("-" * 30)
    
    
    #endregion* Fin Report


    #region* Save
    
    save_root = "/home/strah/Documents/Work_related/thon-of-py/blober/auto_res/"

    for i in range(3):
        sp = save_root + f"strat{i}.pkl"

        with open(sp,"wb") as f:
            pickle.dump(cands[i],f)

    
    #endregion* Save



    #region curr Analysis
    

    #? Moved to seek_feat.py!

    #endregion curr Analysis



    print("STOP")
    print("STOP")

#endregion Main Funcs

if __name__ == "__main__":
    main()
