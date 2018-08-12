# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 11:14:40 2018

@author: 周宝航
"""

import logging
import os.path
import sys
import argparse
from linear_regression import LinearRegression

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    
    parser = argparse.ArgumentParser(prog=program, description = 'train the model by linear regression')
    parser.add_argument("--in_path", "-i", required=True, help="train data path")
    parser.add_argument("--out_path", "-o", help="output model path, file type is : *.pkl")
    parser.add_argument("--num_iters", "-n", type=int,help="iteration times")
    parser.add_argument("--alpha", "-a", type=float, help="learning rate")
    args = parser.parse_args()
    
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    
    lr_model = LinearRegression(num_iters=args.num_iters, alpha=args.alpha)
    logger.info("start training")
    lr_model.train_model(args.in_path)   

    if args.out_path:
        if args.out_path.split('.')[-1] == "pkl":
            lr_model.save(args.out_path)
        else:
            print("model file type error. Please use *.pkl to name your model.")
            sys.exit(1)