import re
import csv
import json
import string
import argparse
import pandas as pd
from tqdm import tqdm
from scipy.special import rel_entr
from collections import OrderedDict


def distribution_count(out_dir, f_name, lower_limit, upper_limit):

    f_data = [json.loads(lines) for lines in open(out_dir + f_name, "r")]

    dict_org = f_data[0]
    #print(dict_org)
    dict_sorted = dict(OrderedDict(sorted(dict_org.items(), key = lambda x : x[1], reverse = True)))

    list_sent_val = []
    for k, v in dict_sorted.items():
        list_sent_val.append(v)
    

    count = 0

    for elem in list_sent_val:
        if lower_limit <= elem <= upper_limit:
            count += 1
    return count

def distribution(L_new, dir, count):
    Dist = []
    for i in range(len(L_new)):
        y = distribution_count(dir, "global_sensitivity_finalvalues.json", L_new[i][0], L_new[i][1])
        Dist.append(y/count)
    return Dist
    

def main():
    parser = argparse.ArgumentParser(description = "Process multiple files")
    parser.add_argument("--dir_1", nargs="+")
    parser.add_argument("--dir_2", nargs="+")
    
    args = parser.parse_args()

    dist_12, dist_21 = [], []
    for dir_1, dir_2 in zip(args.dir_1, args.dir_2):
        print(f"Directory 1: {dir_1}, Directory 2: {dir_2}")
        L_new = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1)]

        count_1 = distribution_count(dir_1, "global_sensitivity_finalvalues.json", 0, 1)
        count_2 = distribution_count(dir_2, "global_sensitivity_finalvalues.json", 0, 1)
        print("count 1 is {} and count 2 is {}".format(count_1, count_2))
        Dist_1 = distribution(L_new, dir_1, count_1)
        Dist_2 = distribution(L_new, dir_2, count_2)
        
        epsilon = 1e-10
        Dist_1 = Dist_1 + epsilon
        Dist_2 = Dist_2 + epsilon


        print(sum(rel_entr(Dist_1, Dist_2)), sum(rel_entr(Dist_2, Dist_1)))
        dist_12.append(sum(rel_entr(Dist_1, Dist_2)))
        dist_21.append(sum(rel_entr(Dist_2, Dist_1)))
        print("*************************************************************************")

    print(dist_12)
    print(dist_21)

    
if __name__ == "__main__":
    main()
    



