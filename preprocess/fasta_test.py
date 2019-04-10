"""description: tests for FASTA commands
"""

import os
import numpy as np
import threading

from fasta import GenomicIntervalConverter


_TEST_FASTA_FILE = "test.fa"


def make_small_fasta_file():
    """make a minimal fasta file to work with
    """
    sequence = """>chr1
aagtgggaatttgtacacccatgtttatagcagcattcacaagagccaaa
gNgtagaaaaagcccaaatctccatctacagatgaatggataagcaaata
tgattgatacacacaatattattcagccttaaaaagaaaagaaattctaa
tNcatgctacaacataaaccttgaaaacattaaaacaaaaaaatgcaaat
actccatgctttccacttttatagtgtacctagagcagttaaattcatag
acacagaaagtaggatgtggtttccaggggatggggggaagggaaagggg
agctattgttccgtggatacagagtttcactttgggatgataacacaggt
ttggaatggatactggtgatagttgcacaacaatgtatttctacttaatg
TACTTAAATTTTCTCTTATACATATTTCACCACAGAAAAAAAATTAGcaa
atatttaaaaatctacatgtttctttgaactgtttctttacgtcccttgt
ccatttttcaatttggctgttggtcttttaataatttaggatagctatat
agctatatatatacatatatattacacaaattagctttttgtctgtcatg
tatgttgcagatctgtttccccagtctgttgacttttgactttgatgcat
ttttcttgcctataaagttttaatttttgtggagtcaaatgcatcagtct
tctATTCAAAGACTtatttatgggtttggtgtttcagtttaggatgaaaa
agaagttgtagatatggataaaggagatggttgtacaacacattatattg
gaccactcaactgtacgtttaaacatggttaaaatggtaagttttatatt
gtatatattttaccaccaaaaaaGGGCCAGGCTTAGATGTTTATATGTTA
GGGGTTTGGAGTACCTCTAACATTTATTCCCCTACAGGGCTCTAATTTAT
AAAACTCTCACACAAGAAAAGTAGATACTGACTTCACAAATCTCCTTACA
AGTCCCACAGCAAGGGCTGTCTGGGAAAGCAGAGGTGGAAAAAGTCACAT
AAACTTGAGATCAGTGTGAGACCTCCCATCCCCCACTCTGGAATCAGATG
GAGGAAGGCAGGCATGCAGGCTGAGCTGGAGAGATGAGCTGGGGTGGGCA
GAACTGTCTTCCCATGAGCCTAGACCTTAAGTGCTCCCACATGATCTCAG
GCATGTATCAAACCAAGAAAGCGGCTAGGAGGGCAACACATCTACCTGTA
TACAGGGAGCTATGAAATATGTGAGCTGCGCAAGTGATGCACAAGGAAAC
GGAAGCAGTATGACCTTTACACAGTGACCTGGCTCAAATAATTTCAGGCT
GTCATTAACCAGGCGAGCTCCACTTTCTCTCTGAGGTAGGTAAAATTGAG
GGGGTAAAGTGGGAGTTGGGGAAAATGGAAAAGAAAGTCTGGTAGTATTT
CTTCTAACTCTGTCATAAATAAAAAGTAAAACATAGATGCCATTTCTCAG
GGCCCAAATGTTAGGTGAAAAAATGTTTGTCATCTCAGTCATGTGATGTG
GACTTCAGCAGAGCAGTACACACATGGTCATTTATCCTTTCCCTCTGCAT"""

    with open(_TEST_FASTA_FILE, "w") as fp:
        fp.write(sequence)
    
    return


def remove_small_fasta_file():
    """remove the test file
    """
    os.system("rm {}*".format(_TEST_FASTA_FILE))
    
    return


def test_converter():
    """make sure converter produces expected output
    """
    # make fasta file
    make_small_fasta_file()

    # make some intervals    
    intervals = np.array([
        "features=chr1:0-10",
        "features=chr1:900-910",
        "features=chr1:150-160"])
    intervals = np.expand_dims(intervals, axis=1)

    # true results (pulled by eye)
    seq1 = [0,0,2,3,2,2,2,0,0,3]
    seq2 = [2,2,2,2,3,3,3,2,2,0]
    seq3 = [3,4,1,0,3,2,1,3,0,1]
    true_results = np.stack([seq1, seq2, seq3], axis=0)
    
    # make a converter
    converter = GenomicIntervalConverter(
        threading.Lock(),
        _TEST_FASTA_FILE,
        intervals.shape[0],
        seq_len=10)

    # convert
    results = converter.convert(intervals, seq_len=10)
    
    # clean up
    converter.close()
    remove_small_fasta_file()
        
    # check
    assert np.array_equal(results, true_results)

    return True
    

    




