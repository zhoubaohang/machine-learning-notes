# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:50:20 2018

@author: 周宝航
"""

import logging
import os.path
import sys
import argparse
from logistic_regression import LogisticRegression
from linear_regression import LinearRegression

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    choices = ["linear", "logistic"]
    parser = argparse.ArgumentParser(prog=program, description = 'train the model by linear regression | logistic regression')
    parser.add_argument("--in_path", "-i", required=True, help="train data path")
    parser.add_argument("--out_path", "-o", help="output model path, file type is : *.pkl")
    parser.add_argument("--num_iters", "-n", type=int,help="iteration times")
    parser.add_argument("--num_features", "-f", type=int,help="number of features selected from data")
    parser.add_argument("--alpha", "-a", type=float, help="learning rate")
    parser.add_argument("--lamb_da", "-l", type=float, help="learning rate")
    parser.add_argument("--type", "-t", required=True, choices=choices, help="choose the model type")
    args = parser.parse_args()
    
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    
    lr_model = None
    if args.type == choices[0]:
        lr_model = LinearRegression(num_iters=args.num_iters, \
                                    num_features=args.num_features, alpha=args.alpha, lamb_da=args.lamb_da)
    elif args.type == choices[1]:
        lr_model = LogisticRegression(num_iters=args.num_iters, \
                                      num_features=args.num_features, alpha=args.alpha, lamb_da=args.lamb_da)
    logger.info("start training")
    lr_model.train_model(args.in_path)   

    if args.out_path:
        if args.out_path.split('.')[-1] == "pkl":
            lr_model.save(args.out_path)
        else:
            print("model file type error. Please use *.pkl to name your model.")
            sys.exit(1)