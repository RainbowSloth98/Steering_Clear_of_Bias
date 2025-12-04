import torch
import pickle



#region Paths

save_p = "/media/strah/344A08F64A08B720/Work_related/avged_acts/"

load_p = "/media/strah/344A08F64A08B720/Work_related/INLP_saves/"

tpicklens_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/INLP/loads/tpicklens.pkl"

#endregion


#region Main Code


#region* Setup

with open(tpicklens_p,"rb") as f:
    doc_lengths = pickle.load(f)

batch_file_indices = [(i * 10) - 1 for i in range(1, 3014)]
file_to_load_idx = 0  # Index for which file to load next

loaded_chunks = torch.tensor([]) # Start with an empty buffer to hold chunks

#endregion


print("STOP")
print("STOP")

# --- Main Loop ---
# Loop through each document's length
for doc_idx, current_doc_len in enumerate(doc_lengths):

    #region* 1. Ensure buffer has enough chunks for the current document
    #? This loop will run as many times as needed (0, 1, or more)

    while len(loaded_chunks) < current_doc_len:
        
        # Check if we've run out of files to load
        if file_to_load_idx >= len(batch_file_indices):
            print("Error: Not enough chunks in files to process all documents.")
            break
        
        # Load the next file of chunks
        next_file_num = batch_file_indices[file_to_load_idx]
        print(f"Buffer has {len(loaded_chunks)} chunks, need {current_doc_len}. Loading file {next_file_num}...")
        
        new_chunks = torch.load(load_p+f"nullspace_acts{next_file_num}.pt")
        
        # Add the new chunks to our buffer
        loaded_chunks = torch.cat([loaded_chunks, new_chunks])
        file_to_load_idx += 1 # Point to the next file for the next time
    
    # Break the outer loop if the inner loop failed
    if len(loaded_chunks) < current_doc_len:
        break

    #endregion

    #* 2. Process the document
    #? Take the chunks for this document from the start of the buffer
    doc_chunks = loaded_chunks[:current_doc_len]
    
    #* 3. Update the buffer by removing the chunks we just used
    loaded_chunks = loaded_chunks[current_doc_len:]
    
    #* 4. Average and Save
    averaged_activations = torch.mean(doc_chunks, dim=0) # Average across the chunks
    torch.save(averaged_activations, save_p+f"averaged_doc{doc_idx}.pt")
    
    print(f"Processed and saved document {doc_idx}. Chunks remaining in buffer: {len(loaded_chunks)}")


#endregion



