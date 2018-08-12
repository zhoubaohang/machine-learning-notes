# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 22:28:54 2018

@author: 周宝航
"""

import logging
import os.path
import sys
import argparse
from neural_network import NeuralNetwork

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    
    parser = argparse.ArgumentParser(prog=program, description = 'train the model by neural network')
    parser.add_argument("--in_path", "-i", required=True, help="train data path")
    parser.add_argument("--out_path", "-o", help="output model path, file type is : *.pkl")
    parser.add_argument("--num_iters", "-n", type=int,help="iteration times")
    parser.add_argument("--alpha", "-a", type=float, help="learning rate")
    parser.add_argument("--regular", "-r", type=int, help="regularization number")
    parser.add_argument("--layers", "-l", help="neural network architechtures, e.g. 784,100,10")
    args = parser.parse_args()
    
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    
    if args.layers:
        sizes = [int(layer) for layer in args.layers.split(',')]
        nn = NeuralNetwork(sizes=sizes, num_iters=args.num_iters, alpha=args.alpha, lam_bda=args.regular)
        logger.info("start training")
        nn.train_model(args.in_path)   

        if args.out_path:
            if args.out_path.split('.')[-1] == "pkl":
                nn.save(args.out_path)
            else:
                print("model file type error. Please use *.pkl to name your model.")
                sys.exit(1)
    else:
        print("Please give the neural network architechtures args as the note of help")