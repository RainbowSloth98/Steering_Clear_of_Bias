#Standard imports
from datasets import load_dataset
import pandas as pd
import pickle
import os
from tqdm import tqdm
import re
import flashtext as ft


#? Info
#region
#?Find and isolate documents that contain protected characteristics
#?Ideally, this should be done by category.
#?


#TODO For some of the categories, we might want to replace activations (man -> woman)
#TODO and for others it would potentially be removing them all together (Black -> _), rather than their counterpoint


#endregion





#* Paths
#region

d_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/Asylex_Dataset/"
nmr = "our_train.pkl"
ds_p = d_p + nmr

s_p = d_p + "new/"
fs_p = s_p + nmr


#endregion  



#* Init defs
#region


#Gender related groupings
sex_m_list = ["man", "men", "boy", "boys", "male", "gentleman", "gentlemen", "father", "husband", "brother", "uncle", "nephew", "son", "actor", "waiter", "steward", "host", "chairman", "spokesman", "congressman", "businessman", "salesman", "forefather", "godfather", "godson", "sir", "mr", "he", "his", "him", "himself", "patriarch", "bachelor", "groom", "widower", "monk", "landlord", "master", "dad", "fiance", "breadwinner", "provider", "head of household", "aggressor", "perpetrator", "fighter", "combatant", "deserter", "malingerer", "effeminate", "sole provider", "masculine", "unmanly", "coward"]
sex_f_list = ["woman", "women", "girl", "girls", "female", "lady", "ladies", "mother", "wife", "sister", "aunt", "niece", "daughter", "actress", "waitress", "stewardess", "hostess", "businesswoman", "foremother", "godmother", "goddaughter", "saleswoman", "congresswoman", "spokeswoman", "chairwoman", "madam", "ma'am", "mrs", "ms", "miss", "she", "her", "hers", "herself", "matriarch", "bachelorette", "bride", "widow", "spinster", "nun", "landlady", "mistress", "maid", "housewife", "mom", "mum", "fiancee", "victim", "survivor", "caregiver", "homemaker", "dependant", "hysterical", "emotional", "promiscuous", "chaste", "vulnerable", "gossip", "nag", "housekeeper", "unwed mother", "single mother"]

#Age related groups
age_y_list = ["teen", "teenager", "adolescent", "youth", "youngster", "minor", "underage", "young adult", "coming-of-age", "child", "children", "infant", "baby", "toddler", "juvenile", "orphan", "dependant", "school-age", "impressionable", "immature", "vulnerable", "of tender years"]
age_m_list = ["middle-aged", "adult", "grown-up", "parent", "working-age", "able-bodied", "prime of life", "mid-life", "mature", "established", "productive", "provider", "breadwinner", "in their prime"]
age_o_list = ["old age", "golden years", "retirement age", "geriatric", "centenarian", "senior", "elder", "elderly", "senior citizen", "pensioner", "retiree", "of advanced age", "frail", "infirm", "vulnerable", "senile", "past working age", "dependent", "fragile"]

#Nationality related groupings
nat_5_list= ["American", "Australian", "British", "Canadian", "New Zealander", "Kiwi", "French", "German", "Italian", "Japanese", "South Korean", "Spanish", "Spaniard", "Swedish", "Swede", "Norwegian", "Danish", "Dane", "Dutch", "Swiss", "Austrian", "Belgian", "Finnish", "Irish", "Icelander", "Portuguese"]
nat_4_list= ["Argentine", "Argentinian", "Brazilian", "Chilean", "Colombian", "Costa Rican", "Greek", "Hungarian", "Israeli", "Lithuanian", "Estonian", "Latvian", "Polish", "Pole", "Czech", "Slovak", "Romanian", "Bulgarian", "Croatian", "Croat", "Maltese", "Singaporean", "Taiwanese", "Uruguayan", "Mexican"]
nat_3_list= ["Bangladeshi", "Bolivian", "Bosnian", "Cambodian", "Ecuadorian", "Egyptian", "Emirati", "Filipino", "Filipina", "Ghanaian", "Guatemalan", "Honduran", "Indian", "Indonesian", "Jordanian", "Kazakh", "Kenyan", "Kuwaiti", "Lebanese", "Malaysian", "Moroccan", "Nepalese", "Nigerian", "Pakistani", "Panamanian", "Paraguayan", "Peruvian", "Qatari", "Saudi", "Senegalese", "Serbian", "Serb", "South African", "Sri Lankan", "Tanzanian", "Thai", "Togolese", "Tunisian", "Turkish", "Turk", "Ugandan", "Uzbek", "Vietnamese", "Zambian", "Zimbabwean"]
nat_2_list= ["Afghan", "Algerian", "Angolan", "Cameroonian", "Chinese", "Congolese", "Eritrean", "Ethiopian", "Haitian", "Iraqi","Libyan", "Malian", "Nicaraguan", "Rwandan", "Salvadoran", "Somali", "Sudanese", "Venezuelan", "Yemeni"]
nat_1_list= ["Russian", "North Korean", "Syrian", "Iranian", "Persian", "Cuban"] # Note: Some overlap with Cat 2 is intentional and reflects the complexity.
nats_list = [nat_5_list,nat_4_list,nat_3_list,nat_2_list,nat_1_list]


disease_list = ["ADHD","AIDS","Albinism","Alcoholism","Allergies","Alzheimer's","Amputation","Amyloidosis","Ankylosing Spondylitis","Anxiety","Aphasia","Arthritis","Asperger's","Asthma","Ataxia","Autism","BDD","Bipolar Disorder","Blindness","Blood Clot","Body Dysmorphic Disorder","Borderline Personality Disorder","Brain Injury","Bulimia","Cancer","Cerebral Palsy","Chronic Fatigue Syndrome","Chronic Pain","Cirrhosis","Cleft Palate","Coeliac Disease","Concussion","COPD","Crohn's Disease","Cystic Fibrosis","Deafness","Deep Vein Thrombosis","Dementia","Depression","Diabetes","Down Syndrome","Dwarfism","Duchenne Muscular Dystrophy","Dyscalculia","Dysgraphia","Dyslexia","Dyspraxia","Dystonia","Eating Disorder","Eczema","Edwards Syndrome","Ehlers-Danlos","Endometriosis","Epilepsy","Fetal Alcohol Spectrum Disorders","Fibromyalgia","Fragile X Syndrome","Gastrointestinal Disorder","Glaucoma","Gout","Hard of Hearing","Hearing Loss","Heart Disease","Hemophilia","Hepatitis","HIV","Huntington's Disease","Hyperactivity","Hypersomnia","Hypertension","Intellectual Disability","Interstitial Cystitis","Irritable Bowel Syndrome","Kernicterus","Kidney Failure","Learning Disability","Leukemia","Lupus","Macular Degeneration","ME/CFS","Meniere's Disease","Mental Illness","Migraine","Mobility Impairment","Motor Neurone Disease","Multiple Sclerosis","Muscular Dystrophy","Myasthenia Gravis","Narcolepsy","Neurocognitive Disorder","Obesity","OCD","Obsessive-Compulsive Disorder","Orthopedic Impairment","Osteoarthritis","Osteoporosis","Panic Disorder","Paralysis","Parkinson's Disease","Personality Disorder","Phobia","Poliomyelitis","Polycystic Ovary Syndrome","Post-Traumatic Stress Disorder","PTSD","Psychiatric Disability","Pulmonary Fibrosis","Rheumatoid Arthritis","Schizophrenia","Sciatica","Scleroderma","Scoliosis","Seizure Disorder","Sepsis","Sickle Cell Disease","Sjogren's Syndrome","Sleep Apnea","Spina Bifida","Spinal Cord Injury","Stammer","Stutter","Stroke","Thalassemia","Thyroid Disorder","Tourette Syndrome","Traumatic Brain Injury","TBI","Tuberculosis","Ulcerative Colitis","Vision Impairment","Von Willebrand Disease"]
disability_list = ["accessible","assistive technology","audiologist","Braille","cane","catheter","cochlear implant","crutches","disability","disabled","handicapped","guide dog","service animal","hearing aid","impairment","inclusion","large print","lip-reading","mobility aid","occupational therapy","personal care assistant","physical therapy","prosthesis","prosthetic","ramp","reasonable accommodation","scooter","screen reader","sign language","special education","speech therapy","support animal","text-to-speech","walker","wheelchair","non-verbal","non-speaking","limited mobility","hard of hearing","partially sighted","neurodivergent","neurodiverse","sensory sensitivity"]

race_list = ["White","Black","African American","Asian","American Indian","Alaska Native","Native Hawaiian","Pacific Islander","Hispanic","Latino","Latina","Latinx","Middle Eastern","North African","MENA","Caucasian","Mongoloid","Negroid","Australoid","Amerindian","Indigenous","Aboriginal","Afro-Caribbean","Afro-Latino","Asian American","Black British","people of color","POC","BIPOC"]
tribe_list = ["Navajo","Cherokee","Sioux","Chippewa","Choctaw","Apache","Lumbee","Pueblo","Iroquois","Creek","Blackfeet","Chickasaw","Comanche","Inuit","Yaqui","Aztec","Maya","Tohono O'odham","Maasai","Sami","Ainu","Quechua","Mapuche","Guarani"]

#Quickly checks the existence of words in text.
kwp = ft.KeywordProcessor()




#endregion  


#* Funcs
#region

def get_df_mask(txt_check,txt_l,kwp = kwp):

    kwp = ft.KeywordProcessor()

    _ = list(map(kwp.add_keyword,txt_l))

    mask = [True if kwp.extract_keywords(txt_check["raw_txt"].loc[i]) else False for i in range(len(txt_check)) ]


    return mask

#Not really useful, but will keep
def _apply_mask(ds,mask):
    return ds[mask]

#endregion




#* Dataset
#region

with open(ds_p,"rb") as f:
    ds = pickle.load(f)
nds = ds.reset_index()
#endregion


#? Ran only once - glacially slow 
#region

# sex_ms = [get_df_mask(nds,sex_m_list),get_df_mask(nds,sex_f_list)]
# age_ms = [get_df_mask(nds,age_y_list),get_df_mask(nds,age_m_list),get_df_mask(nds,age_o_list)]
# nat_ms = [get_df_mask(nds,nat) for nat in nats_list]
# disease_m = get_df_mask(nds,disease_list)
# disability_m = get_df_mask(nds,disability_list)
# race_m = get_df_mask(nds,race_list)

#endregion


#* Instead loading in from pickle
#region

#path defs
#region

m_p = "/home/strah/Desktop/Work_stuff/Papers/2.1_Paper/Code/data/Asylex_Dataset/masks/"

age_ns = ["agey.pkl","agem.pkl","ageo.pkl"]
nat_ns = [f"nat{i}.pkl" for i in range(1,6)]
sex_ns = ["sexm.pkl","sexf.pkl"]
race_ns = "race.pkl"
disease_ns = "disease.pkl"
disability_ns = "disability.pkl"

#endregion

age_ms = []
for nmr in age_ns:
    full_p = m_p + nmr

    with open(full_p,"rb") as f:
        age_ms.append(pickle.load(f))

nat_ms = []
for nmr in nat_ns:
    full_p = m_p + nmr

    with open(full_p,"rb") as f:
        nat_ms.append(pickle.load(f))

sex_ms = []
for nmr in sex_ns:
    full_p = m_p + nmr

    with open(full_p,"rb") as f:
        sex_ms.append(pickle.load(f))

race_p = m_p + race_ns
with open(race_p,"rb") as f:
    race_m = pickle.load(f)

disease_p = m_p + disease_ns
with open(disease_p,"rb") as f:
    disease_m = pickle.load(f)

disability_p = m_p + disability_ns
with open(disability_p,"rb") as f:
    disability_m = pickle.load(f)

#endregion


#Indicies of all true examples
ts = {
    "sex_m": [i for i,v in enumerate(sex_ms[0]) if v == True],
    "sex_f": [i for i,v in enumerate(sex_ms[1]) if v == True],
    "age_y": [i for i,v in enumerate(age_ms[0]) if v == True],
    "age_m": [i for i,v in enumerate(age_ms[1]) if v == True],
    "age_o": [i for i,v in enumerate(age_ms[2]) if v == True],
    "race": [i for i,v in enumerate(race_m) if v == True],
    "nat1": [i for i,v in enumerate(nat_ms[4]) if v == True],
    "nat2": [i for i,v in enumerate(nat_ms[3]) if v == True],
    "nat3": [i for i,v in enumerate(nat_ms[2]) if v == True],
    "nat4": [i for i,v in enumerate(nat_ms[1]) if v == True],
    "nat5": [i for i,v in enumerate(nat_ms[0]) if v == True],
    "disease": [i for i,v in enumerate(disease_m) if v == True],
    "disability": [i for i,v in enumerate(disability_m) if v == True],
}

#Indicies of all false examples
fs = {
    "sex_m": [i for i,v in enumerate(sex_ms[0]) if v == False],
    "sex_f": [i for i,v in enumerate(sex_ms[1]) if v == False],
    "age_y": [i for i,v in enumerate(age_ms[0]) if v == False],
    "age_m": [i for i,v in enumerate(age_ms[1]) if v == False],
    "age_o": [i for i,v in enumerate(age_ms[2]) if v == False],
    "race": [i for i,v in enumerate(race_m) if v == False],
    "nat1": [i for i,v in enumerate(nats_list[4]) if v == False],
    "nat2": [i for i,v in enumerate(nats_list[3]) if v == False],
    "nat3": [i for i,v in enumerate(nats_list[2]) if v == False],
    "nat4": [i for i,v in enumerate(nats_list[1]) if v == False],
    "nat5": [i for i,v in enumerate(nats_list[0]) if v == False],
    "disease": [i for i,v in enumerate(disease_m) if v == False],
    "disability": [i for i,v in enumerate(disability_m) if v == False],
}




#? Main Code
#region






print("STOP")
print("STOP")


#endregion


