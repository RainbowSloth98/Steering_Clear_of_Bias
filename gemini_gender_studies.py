import os
import pandas as pd
import pickle
from tenacity import retry, stop_after_attempt, wait_random_exponential

import vertexai
from vertexai.generative_models import GenerativeModel

import torch
from torch.utils.data import TensorDataset
from collections import Counter
from dotenv import load_dotenv
import time




#region Defs/Inits

# Load the environment variables from the .env file
load_dotenv()

    #? Delay reasoning
    #region
    # Generally max requests are 2000 per min, ~33 per sec, or 1 every 0.03.
    # However, there is also a token limit per min TPM of 4mil/min, or 66k/s.
        #We take our docs to have abt 7k tokens, implying a max of 66k/7k = 9.5 docs pers sec, 
        # or one every 0.12; We set our limit to be 0.15 per doc.
    #endregion


#? Bad system, sidestepping
stage = "other"
# stage = "clean"
# stage = "processing"



#pivot defs/processing
if (stage == "processing"):
    SLEEP_DELAY = 2 #Ultimately went for this, despite above
    FRESH_START = True



#endregion  


#region Funcs

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def gen_with_backoff(model, prompt):
    return model.generate_content(prompt)

#endregion



#region Paths

root_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/genderer/"
save_p = root_p + "step_saves/"
ds_p = root_p + "our_train.pkl"

res_p = root_p + "results/"
res_namer = "clean_gen_df.pkl"
full_res_p = res_p + res_namer

#pivot paths/clean
if(stage=="clean"):
    anal_res_p = save_p + "save_res_last.pkl"
    anal_prob_p = save_p + "save_prob_last.pkl"

    chunk_tok_df_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/Asylex_Dataset/pre_gender/chunked_tokenised_df.pkl"


#endregion



#region Loading...

with open(ds_p,"rb") as f:
    ds = pickle.load(f)




#pivot loading/clean

if(stage=="clean"):
    with open(anal_res_p,"rb") as f:
        anal_res = pickle.load(f)

    with open(anal_prob_p,"rb") as f:
        anal_prob= pickle.load(f)


#endregion  



#pivot Generating Dataset via Gemini
if(stage == "processing"):

    #region Define the model

    try:
        PROJECT_ID = "asylex-gender-extraction"
        LOCATION = "europe-west2"
        vertexai.init(project=PROJECT_ID, location=LOCATION)

    except Exception as e:
        print(f"Error initializing Vertex AI. Have you set your Project ID? Error: {e}")
        exit()

    # The Google library will automatically find the credentials since the
    # GOOGLE_APPLICATION_CREDENTIALS environment variable is now set.
    model = GenerativeModel(model_name="gemini-1.5-flash")

    #endregion  



    #Define the prompt template
    prompt_template = """
    Your task is to determine the gender of the main appellant in the following anonymised legal document, starting at [1].

    **Instructions:**
    1.  Identify the person referred to as " the appellant" or "the claimant", focusing on the main one if there are multiple.
    2.  Carefully analyse the surrounding context for gender specific words (e.g. pronouns) used to refer to this individual.
    3.  Provide your answer in one of the following formats ONLY: Male, Female, or Unclear.

    **Document:**
    ---
    {document_text}
    ---

    **Analysis Result:**
    """



    #region Fresh start options

    if(FRESH_START):

        results = []
        problem_cases = []
        start_index = 0 


    else: #Load the old one.
        preload_res_p = save_p + "save_res_last.pkl"
        preload_prob_p = save_p + "save_prob_last.pkl"

        with open(preload_res_p,"rb") as f:
            results = pickle.load(f)

        with open(preload_prob_p,"rb") as f:
            problem_cases = pickle.load(f)

        #?This might be better done as results[-1][]
        start_index = len(results)

    #endregion



    #region Calling the API to find the gender info



    save_steps = list(range(0,23000,250)) #?Lowered from 1000 to 250 because if we don't save if we error. Now we save more.
    save_steps.append(len(ds)) #The final one will serve as results
    save_steps = set(save_steps)

    # Loop through each row of the DataFrame
    for index, row in ds.iterrows():
        if(index >= start_index):


            # Make sure your CSV has a column named 'document_text'
            document_text = row['raw_txt']

            # Format the prompt with the text from the current row
            prompt = prompt_template.format(document_text=document_text)


            try:

                print(f"Processing document {index + 1}/{len(ds)}...")
                response = gen_with_backoff(model,prompt)


                results.append({"i":index,"resp":response.text.strip()})



            except Exception as e:
                problem_cases.append({"i":index,"txt":document_text,"err":str(e)})
                continue #Skip the example and continue along

            #Saving
            if(index in save_steps):
                step_res_p = save_p + f"save_res_{index}.pkl"
                last_res_p = save_p + "save_res_last.pkl"

                step_prob_p = save_p + f"save_prob_{index}.pkl"
                last_prob_p = save_p + "save_prob_last.pkl"

                with open(step_res_p,"wb") as f:
                    pickle.dump(results,f)
            
                with open(last_res_p,"wb") as f:
                    pickle.dump(results,f)

                with open(step_prob_p,"wb") as f:
                    pickle.dump(problem_cases,f)
            
                with open(last_prob_p,"wb") as f:
                    pickle.dump(problem_cases,f)

            # Respect API rate limits
            time.sleep(SLEEP_DELAY)

        #If index<start_index
        else:
            pass # Do nothing, we want to continue where we left off.

    #endregion  



#pivot Cleaning Dataset
if (stage == "clean"):

    #? Cleaned out the dataset from bad entries
    # indies = [x["i"] for x in anal_res]
    # fin_ds = ds.loc[indies]
    # fin_ds = fin_ds.drop(columns=["index"])
    # with open(full_res_p,"wb") as f:
    #     pickle.dump(fin_ds,f)


    #? Now we remove the entries that aren't associated 
    # with open(full_res_p,"rb") as f:
    #     gender_dat = pickle.load(f)

    # with open(chunk_tok_df_p,"rb") as f:
    #     chunk_tok_df= pickle.load(f)


    #? Remove all of the unclears - we saved this as mf_gen_df (genderer/results)
    # uncs = [i for i,x in enumerate(anal_res) if x["resp"] == "Unclear"]
    # mf_gendat = gender_dat.drop(uncs)


    #? Cleaning the .explode() chunked tokenized one as well - saved as post_genderer/mf_ct_df
    # #Check the indicies that were kept by gender_dat
    # indies = list(gender_dat.index)

    # #Find a list of those that were not
    # dropdex = []
    # for i,r in chunk_tok_df.iterrows():
    #     if not(r["og_doc_id"] in indies):
    #         dropdex.append(i)
    # #remove them
    # ndf = chunk_tok_df.drop(dropdex)


    #region# Gendered DF to TensorDataset (Original, Deprecated)
    
    
    # # Loading Gender ds
    # mf_ct_df_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/Asylex_Dataset/post_gender/mf_ct_df.pkl"
    # with open(mf_ct_df_p,"rb") as f:
    #     mf_ct_df= pickle.load(f)


    # #? Padding - saved as post_gender/post_pad_df.pkl
    # for i,r in mf_ct_df.iterrows():
    #     if(len(r["chunks"]) < 512):
    #         r["chunks"].extend([0]*(512-len(r["chunks"])))

    # #Hacky because we first saved this
    # post_pad_df = mf_ct_df

    # #Create a tensor out of the df - make sure it uses dtype int64 for BERT
    # mf_tens = torch.Tensor(list(post_pad_df["chunks"])).to(torch.int64)

    # #Create a mask for the tensor - make sure it uses dtype long
    # atten_mask = (mf_tens != 0).to(torch.long)

    # #Using this, create a TensorDataset
    # tens_ds_obj = TensorDataset(mf_tens,atten_mask)

    # #Saveing
    # mf_ds_obj_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/Asylex_Dataset/post_gender/mf_tens_ds_obj.pt"
    # with open(mf_ds_obj_p,"wb") as f:
    #     torch.save(tens_ds_obj,f)

    #endregion

    print("STOP")
    print("STOP")




#region# Creating gendered dataframe 26.9.25


# #? Load stuff first
# our_train_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/Asylex_Dataset/base_clean/our_train.pkl"
# prob_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/genderer/step_saves/save_prob_last.pkl"
# res_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/genderer/step_saves/save_res_last.pkl"
# lab_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/Asylex_Dataset/base_clean/train_labels_23121.pt"


# out_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/genderer/results/cleargen_train_26_9_25.pkl"
# labsafe = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/genderer/results/gen_labs.pkl"

# with open(our_train_p,"rb") as f:
#     our_train = pickle.load(f)

# with open(prob_p,"rb") as f:
#     prob = pickle.load(f)

# with open(res_p,"rb") as f:
#     res = pickle.load(f)

# with open(lab_p,"rb") as f:
#     labs = torch.load(f)

# prob_s = {prob[i]["i"] for i in range(len(prob))} #* Make a set out of the problem rows
# noprob_train = our_train.drop(prob_s) #* Remove them from df
# more23121 = list(range(23122,23144)) #* Make a list of indices that don't appear
# gen_train = noprob_train.drop(more23121) #* remove those too

# # * Get the gender info into a form that is easily added to the df
# gen_info = [k["resp"] for k in res]
# gen_dict = {"subject_gender":gen_info}

# gen_train.insert(0,"subject_gender",gen_dict["subject_gender"]) #* add it there

# cleargen_train = gen_train[gen_train.subject_gender != "Unclear"] #* Remove the unclear entries

# ri_cleargen_train = cleargen_train.reset_index() #* Reset the index for ease

# # * Save
# # with open(out_p,"wb") as f:
# #     pickle.dump(ri_cleargen_train,f)


# #? Now the labels!

# lastrem_labs = labs[:23099] #* Total of 22 above 23121 to remove (org max was 23143)

# #* Create a tensor of the probs and unclears to remove
# unc_indies = list(gen_train[gen_train.subject_gender == "Unclear"].index)
# prob_l = list(prob_s)
# rem_indies = unc_indies + prob_l
# t_rem_indies = torch.tensor(rem_indies)

# #* Use tensor to make a mask, and apply mask to remove indices
# mask = torch.ones(lastrem_labs.size())

# mask[t_rem_indies] = False

# mask = mask.to(torch.bool) #! For this to work, the dtype *must* be bool.

# cleargen_labs = lastrem_labs[mask]


# with open(labsafe,"wb") as f:
#     pickle.dump(cleargen_labs,f)


#endregion


#region curr Dataframe to Dataset 30.9.25

cleargen_labs_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/Asylex_Dataset/gendered_clean/cleargen_labs_26_9_25.pkl"
cleargen_df_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/Asylex_Dataset/gendered_clean/cleargen_train_26_9_25.pkl"

with open(cleargen_labs_p,"rb") as f:
    cleargen_labs= pickle.load(f)

with open(cleargen_df_p,"rb") as f:
    cleargen_df= pickle.load(f)






print("STOP")
print("STOP")
#endregion





print("FILE END")