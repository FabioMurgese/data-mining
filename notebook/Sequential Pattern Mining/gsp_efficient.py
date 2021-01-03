import gsp as gsp
import pandas as pd
#from fim import apriori

img = 'plots/figure_{}.png'
dataFolder = '../../data/{}'
import subprocess
import datetime
import tqdm

def load_dataset():
    df = pd.read_csv(dataFolder.format('customer_supermarket.csv'), sep='\t', index_col=0).drop_duplicates()

    df['Sale'] = df['Sale'].str.replace(",", ".").astype(float)  # replace ',' with '.' to make Sale type as float64
    df['CustomerID'] = df['CustomerID'].astype('Int64')
    df['BasketDate'] = pd.to_datetime(df['BasketDate'])
    df = df[df['CustomerID'].notna()]
    return df


def load_customer_baskets():
    df = load_dataset()

    df['ProdID'] = df['ProdID'].astype('category')

    # Create a dictionary of ProdID (key) and its mapping to an integer as value
    prod_n_mapping = dict(zip(df['ProdID'], df['ProdID'].cat.codes))

    # Create a dictionary of mapped ProdID to int (key) and its description as value
    prod_cat_descr = dict(zip(df['ProdID'].cat.codes, df['ProdDescr']))

    df = df[['CustomerID', 'ProdID', 'BasketID', 'BasketDate']]
    df = df.sort_values(by='BasketDate')
    df = df.groupby(['CustomerID', 'BasketID'])['ProdID'].apply(list).reset_index()
    df = df.groupby('CustomerID')['ProdID'].apply(list).reset_index()
    customer_baskets = []
    for index, row in df.iterrows():
        customer_baskets.append({
            'customer_id': row['CustomerID'],
            'baskets': row['ProdID']
        })
    return customer_baskets, prod_n_mapping, prod_cat_descr




"""
The input file format is defined as follows. It is a text file where each line represents a sequence
from a sequence database. Each item from a sequence is a positive integer and items from the same 
itemset within a sequence are separated by single space. Note that it is assumed that items within a
same itemset are sorted according to a total order and that no item can appear twice in the same itemset. 
  
The value "-1" indicates the end of an itemset. 
The value "-2" indicates the end of a sequence (it appears at the end of each line). 

For example, the input file "contextPrefixSpan.txt" contains the following four lines (four sequences).

    1 -1 1 2 3 -1 1 3 -1 4 -1 3 6 -1 -2
    1 4 -1 3 -1 2 3 -1 1 5 -1 -2
    5 6 -1 1 2 -1 4 6 -1 3 -1 2 -1 -2
    5 -1 7 -1 1 6 -1 3 -1 2 -1 3 -1 -2

The first line represents a sequence where the itemset {1} is followed by the itemset {1, 2, 3}, followed by the itemset {1, 3}, followed by the itemset {4}, followed by the itemset {3, 6}. The next lines follow the same format.


@CONVERTED_FROM_TEXT
@ITEM=1=apple
@ITEM=2=orange
@ITEM=3=tomato
@ITEM=4=milk
@ITEM=5=bread
@ITEM=6=noodle
@ITEM=7=rice
@ITEM=-1=|

"""


import os
if not os.path.isfile('input.txt'):

    gsp_dataset, prod_n_mapping, prod_cat_descr = load_customer_baskets()
    dataset = [c['baskets'] for c in gsp_dataset]

    # This is to save the mapping between the ProdIDs and their Descriptions
    # in order to have a textual description in the end
    with open('input.txt', 'w') as f:
        f.write('@CONVERTED_FROM_TEXT\n')
        f.write('@ITEM=-1=|\n')

        for k, v in prod_cat_descr.items():
            f.write('@ITEM={}={}\n'.format(k, v))

    # Create the input.txt file that will be passed to the library
    with open('input.txt', 'a') as f:

        for sequence in tqdm.tqdm(dataset):
            string = ""

            if len(sequence)>1:
                for itemset in sequence:
                    for item in sorted(itemset, key=str.lower):
                       string += "{} ".format(prod_n_mapping[item])
                    string += "-1 " #  The value "-1" indicates the end of an itemset.
                string += "-2" # The value "-2" indicates the end of a sequence (it appears at the end of each line).
                f.write(string+"\n")

support_percentage = "5%"

# This run the process
if not os.path.isfile("results/results_support_{}.txt".format(support_percentage)):

    start = datetime.datetime.now()
    subprocess.call(["java", "-jar", "spmf.jar", "run", "GSP", "input.txt", "results/results_support_{}.txt".format(support_percentage),
                     "{}".format(support_percentage), "false"])
    end = datetime.datetime.now()
    print("Total seconds elapsed {}s".format((end-start).total_seconds()))

# read output rules
lines = []
try:
    with open("results/results_support_{}.txt".format(support_percentage), "rU") as f:
        lines = f.readlines()
except:
    print
    "read_output error"

patterns = []
for line in lines:
    line = line.strip()
    patterns.append(line.split(" -1 "))

n_pattern_found = 0
sequences = []
supp = []

useful = 0
for p in patterns:
    p = p[0].split("|")

    if len(p) > 2:
        useful += 1

        sx_str = ""
        n_elements = len(p)
        for i, el in enumerate(p):
            if i == (n_elements-1):
                el = el.replace("#SUP: ", "")
                supp.append(int(el))

            else:
                if sx_str != "":
                    sx_str += ","+el.strip()
                else:
                    sx_str += el.strip()

        sequences.append(sx_str.strip())
        n_pattern_found += 1

df = pd.DataFrame({'Sequences': sequences, 'Support': supp})
df = df.sort_values(by=['Support'], ascending=False)
print(df[:150].set_index('Sequences').to_markdown())

df.to_csv('results/results_support_{}.csv'.format(support_percentage))


row_with_more_sx = 0
for idx, row in df.iterrows():
    if "," in row['Sequences']:
        splitted = row['Sequences'].split(",")
        if len(set(splitted)) >= 2:
            row_with_more_sx += 1

print("Patterns found: {}".format(n_pattern_found))

