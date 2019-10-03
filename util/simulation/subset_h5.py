import h5py
from Bio import SeqIO

dataset = 'sim_B_2'

sequences = { seq.id: seq for seq in SeqIO.parse("{}/assembly.fasta".format(dataset),"fasta") }
h5file = h5py.File("{}/coverage_contigs.h5".format(dataset))
h5file_sub = h5py.File("{}/coverage_contigs_sub.h5".format(dataset),'w')

seq_filt = []

print("Contigs in fasta: {}".format(len(sequences)))
print("Contigs in h5py: {}".format(len(h5file)))

for ctg_name, seq in sequences.items():
    
    coverage = h5file.get(ctg_name)[:2,:]

    sample_sums = coverage.sum(axis=1)

    if sum(sample_sums>len(seq)) > 0 and len(seq) >= 2048:
        seq_filt.append(seq)
        h5file_sub.create_dataset(ctg_name,data=coverage)

SeqIO.write(seq_filt,"{}/assembly_gt2048.fasta".format(dataset),"fasta")
