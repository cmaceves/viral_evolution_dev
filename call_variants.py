import os
import sys


reference = "/home/chrissy/Desktop/sequence.fasta"

all_bam = [os.path.join("./sorted_bam", x) for x in os.listdir("./sorted_bam") if x.endswith(".bam")]
for bam in all_bam:
    sra_id = bam.split("/")[-1].replace(".sorted.bam","")
    output = "./variants/%s_variants" %sra_id
    cmd = "samtools mpileup -aa -A -d 20000 -B -Q 0 --reference %s  %s | ivar variants -p %s -q 10 -t 0.0001 -m 1 -r %s"%(reference, bam, output, reference)
    os.system(cmd)
