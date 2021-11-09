#!/usr/bin/env python3

from __future__ import print_function

import argparse
import logging
import sys

import json

post = ''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a conversation graph to a set of connected components (i.e. threads).')
    parser.add_argument('--raw_list', help='List of raw text documents containing the raw log content as <filename>:...')
    parser.add_argument('--result', help='File containing the cluster content as <filename>:...')

    args = parser.parse_args()

    train_raw = open('./{}/train_dials.json'.format(args.raw_list))
    dev_raw = open('./{}/dev_dials.json'.format(args.raw_list))
    test_raw = open('./{}/test_dials.json'.format(args.raw_list))

    train_result_i = open('./SnipsNSD5%/train/seq.in'.format(args.result), "w")
    train_result_o = open('./SnipsNSD5%/train/seq.out'.format(args.result), "w")

    dev_result_i = open('./SnipsNSD5%/valid/seq.in'.format(args.result), "w")
    dev_result_o = open('./SnipsNSD5%/valid/seq.out'.format(args.result), "w")

    test_result_i = open('./SnipsNSD5%/test/seq.in'.format(args.result), "w")
    test_result_o = open('./SnipsNSD5%/test/seq.out'.format(args.result), "w")

    train_json = json.load(train_raw)

    for dialog in train_json:
        for turn in dialog["dialogue"]:
            assert len(turn["user"]["transcript"]) == len(turn["user"]["label"])
            # ut_l = ' '.join([ut.strip() for ut in turn["user"]["transcript"]])
            # ul_l = ' '.join([(ul + post).strip() for ul in turn["user"]["label"]])
            ut_l = ""
            ul_l = ""

            for ut, ul, in zip(turn["user"]["transcript"], turn["user"]["label"]):
                if ul == "B" or ul == "I":
                    ul = ul + post
                if ut.strip() != "" and ul.strip() != "":
                    ut_l = ut_l + " " + ut.lower()
                    ul_l = ul_l + " " + ul

            # train_result_i.write("{}\n".format(ut_l))
            # train_result_o.write("{}\n".format(ul_l))


            assert len(turn["agent"]["transcript"]) == len(turn["agent"]["label"])

            # at_l = ' '.join([at.strip() for at in turn["agent"]["transcript"]])
            # al_l = ' '.join([(al + post).strip() for al in turn["agent"]["label"]])
            at_l = ""
            al_l = ""

            for at, al in zip(turn["agent"]["transcript"], turn["agent"]["label"]):
                if al == "B" or al == "I":
                    al = al + post
                if at.strip() != "" and al.strip() != "":
                    at_l = at_l + " " + at.lower()
                    al_l = al_l + " " + al

            train_result_i.write("{} {}\n".format(ut_l, at_l))
            train_result_o.write("{} {}\n".format(ul_l, al_l))
    
    dev_json = json.load(dev_raw)

    for dialog in dev_json:
        for turn in dialog["dialogue"]:
            ut_l = ""
            ul_l = ""
            assert len(turn["user"]["transcript"]) == len(turn["user"]["label"])
            for ut, ul, in zip(turn["user"]["transcript"], turn["user"]["label"]):
                if ul == "B" or ul == "I":
                    ul = ul + post
                if ut.strip() != "" and ul.strip() != "":
                    ut_l = ut_l + " " + ut.lower()
                    ul_l = ul_l + " " + ul

            # dev_result.write("===\n".format(ut, ul))
            at_l = ""
            al_l = ""

            assert len(turn["agent"]["transcript"]) == len(turn["agent"]["label"])
            for at, al in zip(turn["agent"]["transcript"], turn["agent"]["label"]):
                if al == "B" or al == "I":
                    al = al + post
                if at.strip() != "" and al.strip() != "":
                    at_l = at_l + " " + at.lower()
                    al_l = al_l + " " + al

            dev_result_i.write("{} {}\n".format(ut_l, at_l))
            dev_result_o.write("{} {}\n".format(ul_l, al_l))

    test_json = json.load(test_raw)

    for dialog in test_json:
        for turn in dialog["dialogue"]:
            ut_l = ""
            ul_l = ""
            assert len(turn["user"]["transcript"]) == len(turn["user"]["label"])
            for ut, ul, in zip(turn["user"]["transcript"], turn["user"]["label"]):
                if ul == "B" or ul == "I":
                    ul = ul + post
                if ut.strip() != "" and ul.strip() != "":
                    ut_l = ut_l + " " + ut.lower()
                    ul_l = ul_l + " " + ul

            # test_result.write("===\n".format(ut, ul))
            at_l = ""
            al_l = ""

            assert len(turn["agent"]["transcript"]) == len(turn["agent"]["label"])
            for at, al in zip(turn["agent"]["transcript"], turn["agent"]["label"]):
                if al == "B" or al == "I":
                    al = al + post
                if at.strip() != "" and al.strip() != "":
                    at_l = at_l + " " + at.lower()
                    al_l = al_l + " " + al
            
            test_result_i.write("{} {}\n".format(ut_l, at_l))
            test_result_o.write("{} {}\n".format(ul_l, al_l))

