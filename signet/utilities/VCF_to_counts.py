import os
from tkinter.messagebox import NO
import pandas as pd
import pysam
from collections import Counter

from signet import DATA

# https://www.biostars.org/p/334253/
def VCF_to_counts(vcf_path, reference_genome_path):
    # open vcf file
    vcf = pysam.VariantFile(vcf_path)
    # open fasta file
    if reference_genome_path == None:
        print("ERROR: You should provide a path to the reference genome of your data!")
        exit()
    else:
        genome = pysam.FastaFile(reference_genome_path)
    # define by how many bases the variant should be flanked
    flank = 1
    # iterate over each variant
    list_of_mutations = []
    for record in vcf:
        # extract sequence
        #
        # The start position is calculated by subtract the number of bases
        # given by 'flank' from the variant position. The position in the vcf file
        # is 1-based. pysam's fetch() expected 0-base coordinate. That's why we
        # need to subtract on more base.
        #
        # The end position is calculated by adding the number of bases
        # given by 'flank' to the variant position. We also need to add the length
        # of the REF value and subtract again 1 due to the 0-based/1-based thing.
        # 
        # We need to check that 'chr' is contained in the chromosome, otherwise it doesn't work.
        if 'chr' not in record.chrom:
            chr = 'chr' + record.chrom
        else:
            chr = record.chrom
        # Now we have the complete sequence like this:
        # [number of bases given by flank]+REF+[number of bases given by flank]
        seq = genome.fetch(chr, record.pos-1-flank, record.pos-1+len(record.ref)+flank)
        # print out tab seperated columns:
        # CRHOM, POS, REF, ALT, flanking sequencing with variant given in the format '[REF/ALT]'
        mutation = '{}[{}>{}]{}'.format(seq[:flank].upper(), record.ref.upper(), record.alts[0].upper(), seq[flank+len(record.ref):].upper())
        list_of_mutations.append(mutation)
    # Now we count the number of times each mutation type is present in the list
    counts = Counter(list_of_mutations)
    # Filter and sort based on the mutation categories. 
    mutation_type_order=os.path.join(DATA, "mutation_type_order.xlsx")
    mutation_order = pd.read_excel(mutation_type_order)
    muts_sorted = []
    for mutation_type in mutation_order['Type']:
        muts_sorted.append(counts[mutation_type])
    final_dict = {mutation_order.loc[i, 'Type']: [muts_sorted[i]] for i in range(len(mutation_order))}
    final_df = pd.DataFrame(final_dict, index = ['Sample'])
    return final_df

def bed_to_counts(bed_path, reference_genome_path):
    # open vcf file
    bed = pd.read_csv(bed_path, header=0, sep='\t', index_col=False)
    # open fasta file
    if reference_genome_path == None:
        print("ERROR: You should provide a path to the reference genome of your data!")
        exit()
    else:
        genome = pysam.FastaFile(reference_genome_path)
    # define by how many bases the variant should be flanked
    flank = 1
    mutation_type_order=os.path.join(DATA, "mutation_type_order.xlsx")
    mutation_order = pd.read_excel(mutation_type_order)
    final_df = pd.DataFrame(columns=mutation_order['Type'])
    list_of_samples = bed['sample'].unique()
    for sample in list_of_samples:
        bed_sample = bed[bed['sample']==sample]
        # iterate over each variant
        list_of_mutations = []
        for index, row in bed_sample.iterrows():
            # extract sequence
            #
            # The start position is calculated by subtract the number of bases
            # given by 'flank' from the variant position. The position in the bed file
            # is 0-based. pysam's fetch() expected 0-base coordinate. 
            #
            # The end position is calculated by adding the number of bases
            # given by 'flank' to the variant position. We also need to add the length
            # of the REF value.
            # 
            # We need to check that 'chr' is contained in the chromosome, otherwise it doesn't work.
            if 'chr' not in row['chr']:
                chr = 'chr' + row['chr']
            else:
                chr = row['chr']
            # Now we have the complete sequence like this:
            # [number of bases given by flank]+REF+[number of bases given by flank]
            seq = genome.fetch(chr, row['start']-flank, row['start']+len(row['ref'])+flank)
            # print out tab seperated columns:
            # CRHOM, POS, REF, ALT, flanking sequencing with variant given in the format '[REF/ALT]'
            mutation = '{}[{}>{}]{}'.format(seq[:flank].upper(), row['ref'].upper(), row['alt'][0].upper(), seq[flank+len(row['ref']):].upper())
            list_of_mutations.append(mutation)
        # Now we count the number of times each mutation type is present in the list
        counts = Counter(list_of_mutations)
        # Filter and sort based on the mutation categories. 
        muts_sorted = []
        for mutation_type in mutation_order['Type']:
            muts_sorted.append(counts[mutation_type])
        sample_dict = {mutation_order.loc[i, 'Type']: [muts_sorted[i]] for i in range(len(mutation_order))}
        sample_df = pd.DataFrame(sample_dict, index = [sample])
        final_df = pd.concat((final_df, sample_df))
    final_df = final_df.rename_axis(None, axis=1)
    return final_df