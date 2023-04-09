# data preparation for graphs
# Path: streamlit_graphs/data_preparation.py
import pandas as pd
import glob

tag_name = 'covid'
# integrate the data from all csv files
folder_path = f"/Users/xxxibdara/Downloads/733-final-project/scraping/{tag_name}"

# read all csv files in the folder
all_files = glob.glob(folder_path + "/*.csv")

# save them into a dataframes
df_list = []

for file_path in all_files:
    df = pd.read_csv(file_path)
    df_list.append(df)

all_data = pd.concat(df_list, axis=0, ignore_index=True)

# save the data into dataframe
df = pd.DataFrame(all_data)
df.to_csv(f"{tag_name}.csv", index=False)