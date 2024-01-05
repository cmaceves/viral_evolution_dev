import os
import sys
import numpy as np
import pandas as pd 

"""
all_dirs = ["./"+x+"/"+x for x in os.listdir("./") if x.startswith("SRR")]
print(all_dirs)
for bam in all_dirs:
    print(bam)
    cmd = "samtools mpileup -aa -A -d 20000 -B -Q 0 --reference /home/chrissy/Desktop/sequence.fasta %s | ivar variants -p %s_variants -r /home/chrissy/Desktop/sequence.fasta -t 0.001" %(bam, bam)
    os.system(cmd)
    sys.exit(0)
"""

filename = "./ww_test_data/subset_ww.tsv"
df = pd.read_table(filename)
df.drop(["sample_name", "Unnamed: 0.1"], axis=1, inplace=True)
df.rename(columns={"Unnamed: 0":"sample_name"}, inplace=True)
print(df)
df.to_csv("wastewater_ncbi.csv", index=False)
sys.exit(0)
accessions = []
for index, row in df.iterrows():
    accessions.append(row['Unnamed: 0'])

for i,acc in enumerate(accessions):
    if os.path.isdir(acc):
        continue
    print(i/len(accessions))
    string = "aws s3 sync s3://sra-pub-run-odp/sra/{accession} {accession} --no-sign-request".format(accession=acc)
    os.system(string)
    break

"""
filename = "./ww_test_data/all_metadata.csv"
new_filename = "./ww_test_data/subset.csv"
df = pd.read_csv(filename)
df = df[df['geo_loc_name'] == 'USA: Colorado, Denver, county']
df = df[(df['collection_date'] > '2021-11-01') & (df['collection_date'] < '2022-03-01')]
df.to_csv(new_filename)
"""
