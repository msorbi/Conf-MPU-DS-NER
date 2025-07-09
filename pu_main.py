import torch
import os
import argparse
from utils.data_process import DataProcess
from utils.data_process_entity_prob import DataProcessEntityProb
from utils.model_bnpu import BNPU
from utils.model_mpu import MPU
from utils.model_conf_mpu import ConfMPU
from utils.model_mpn import MPN
from utils.base_classes import Trainer
import json


def bnPU(args):
    dp_ = DataProcess(args)
    if 'BC5CDR' in args.dataset:
        model_ = BNPU(dp_, args.cn, inputSize=250)
    else:
        model_ = BNPU(dp_, args.cn)
    return dp_, model_


def mPU(args):
    dp_ = DataProcess(args)
    if 'BC5CDR' in args.dataset:
        model_ = MPU(dp_, args.cn, inputSize=250)
    else:
        model_ = MPU(dp_, args.cn)
    return dp_, model_


def conf_mPU(args):
    dp_ = DataProcessEntityProb(args)
    if 'BC5CDR' in args.dataset:
        model_ = ConfMPU(dp_, args.cn, inputSize=250)
    else:
        model_ = ConfMPU(dp_, args.cn)
    return dp_, model_


def mPN(args):
    dp_ = DataProcess(args)
    if 'BC5CDR' in args.dataset:
        model_ = MPN(dp_, args.cn, inputSize=250)
    else:
        model_ = MPN(dp_, args.cn)
    return dp_, model_


def main():
    parser = argparse.ArgumentParser(description="PU NER")
    parser.add_argument('--type', type=str, default="bnPU", help='learning type (bnPU/bPN/mPU/conf_mPU/mpn)')
    parser.add_argument('--cn', type=int, default=2, help='class number')
    parser.add_argument('--loss', type=str, default='SMAE', help='loss function (SMAE/MAE)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--beta', type=float, default=0.0, help='beta of pu learning (default 0.0)')
    parser.add_argument('--gamma', type=float, default=1.0, help='gamma of pu learning (default 1.0)')
    parser.add_argument('--drop_out', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--m', type=float, default=30, help='class balance rate')
    parser.add_argument('--weights', type=str, default='', help='weights among positive classes')
    parser.add_argument('--eta', type=str, default="0.5", help='threshold for selecting samples')
    parser.add_argument('--dataset', type=str, default="CoNLL2003", help='name of the dataset')
    parser.add_argument('--flag', type=str, default="ALL", help='train.flag.txt')
    parser.add_argument('--suffix', type=str, default='', help='input suffix')
    parser.add_argument('--priors', type=str, help='priors of positive classes')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for training and testing')
    parser.add_argument('--epochs', type=int, default=500, help='epoch number of training')
    parser.add_argument('--print_time', type=int, default=1, help='epochs for printing result')
    parser.add_argument('--pert', type=float, default=1.0, help='percentage of data_backup use for training')
    parser.add_argument('--determine_entity', type=bool, default=False, help='determine entity or not')
    parser.add_argument('--add_probs', type=bool, default=False, help='whether  add the confidence probs into a file with added_flag')
    parser.add_argument('--added_suffix', type=str, default="prob", help='a name to be added as a suffix')
    parser.add_argument('--inference', type=bool, default=False, help='do inference or not')
    parser.add_argument('--trail', type=int, default=1, help='number of trails of an experiment')
    parser.add_argument('--tag2Idx', type=str, help='dictionary --> key: tag, value: idx')
    parser.add_argument('--idx2tag', type=str, help='dictionary --> key: idx, value: tag')
    parser.add_argument('--embedding', type=str, default='glove.6B.100d.txt', help='embedding file name')
    parser.add_argument('--model_name', type=str, help='saved model name')
    parser.add_argument('--early_stop', type=bool, default=False, help='use validation set to early stop')
    parser.add_argument('--no_lexicon', type=bool, default=False, help='without lexicon feature')
    parser.add_argument('--model_path', type=str, default=None, help='model path for add_probs, example: saved_model/bPN_CoNLL2003_Fully_Entity_NA_lr_0.0001_cn_2_loss_SMAE_m_15.0_ws_NA_eta_0.5_percent_1.0_trail_1')

    args = parser.parse_args()
    assert not (args.add_probs and (args.model_path is None or not os.path.exists(args.model_path))) 

    # set specific arguments for different datasets
    if 'CoNLL2003' in args.dataset:
        args.cn = 5
        args.tag2Idx = {"O": 0, "PER": 1, "LOC": 2, "ORG": 3, "MISC": 4}
        args.idx2tag = {0: "O", 1: "PER", 2: "LOC", 3: "ORG", 4: "MISC"}
        if 'Fully' in args.dataset:
            args.priors = [0.05465055176037835, 0.040747270664617107, 0.04923362521547384, 0.02255661253014178]  # true
        else:
            args.priors = [0.0314966102568, 0.0376880632424, 0.0354240324761, 0.015502139428]   # estimated
    elif 'BC5CDR' in args.dataset:
        args.cn = 3
        args.tag2Idx = {"O": 0, "Chemical": 1, "Disease": 2}
        args.idx2tag = {0: "O", 1: "Chemical", 2: "Disease"}
        if 'Fully' in args.dataset:
            args.priors = [0.060108318524160105, 0.060082931370060086]  # true
        else:
            args.priors = [0.0503131404897, 0.0503834263676]  # estimated
    elif args.dataset.lower().startswith("hdsner-"):
        with open(os.path.join("data", args.dataset, "priors.json")) as fp:
            priors_tmp = json.load(fp)
        args.tag2Idx = {x:i for i,x in enumerate(["O"]+list(priors_tmp.keys()))}
        args.idx2tag = {v:k for k,v in args.tag2Idx.items()}
        args.priors = list(priors_tmp.values()) # preserve order
        args.cn = len(args.tag2Idx)
        del priors_tmp
    else:
        raise Exception('Please check the dataset name!')

    # set dp and model for different learning types
    dp = None
    model = None
    if args.type == 'bnPU':
        args.cn = 2
        if args.flag == 'Entity' or args.add_probs:
            args.priors = [sum(args.priors)]
        elif not args.inference:
            args.priors = [args.priors[args.tag2Idx[args.flag] - 1]]
        args.eta = 0
        dp, model = bnPU(args)
    elif args.type == 'bPN':
        args.cn = 2
        if args.flag == 'Entity' or args.add_probs:
            args.priors = [sum(args.priors)]
            dp, model = mPN(args)
    elif args.type == 'mPU':
        args.eta = 0
        dp, model = mPU(args)
    elif args.type == 'conf_mPU':
        dp, model = conf_mPU(args)
    elif args.type == 'mPN':
        args.eta = 0
        dp, model = mPN(args)
    else:
        raise Exception('Please check the PU learning type!')

    args.model_name = "{}_{}_{}_{}_lr_{}_cn_{}_loss_{}_m_{}_ws_{}_eta_{}_percent_{}_trail_{}".format(
        args.type,
        args.dataset,
        args.flag,
        args.suffix if args.suffix != '' else "NA",
        args.lr,
        args.cn,
        args.loss,
        args.m,
        args.weights if 'mPU' in args.type else "NA",
        args.eta if args.eta != 0 else "NA",
        args.pert,
        args.trail)

    if torch.cuda.is_available():
        model.cuda()
        torch.cuda.manual_seed(1013)

    # set trainer
    trainer = Trainer(model, args.lr)

    # inference for bnPU
    if args.inference:
        testSet = dp.load_testset(args.dataset, "test.txt", args.no_lexicon)
        # manually pass the model path
        trainer.saved_models = [
            "saved_model/{}_{}_{}_{}_lr_{}_cn_{}_loss_{}_m_{}_ws_{}_eta_{}_percent_{}_trail_{}".format(
                args.type,
                args.dataset,
                k,
                args.suffix if args.suffix != '' else "NA",
                args.lr,
                args.cn,
                args.loss,
                args.m,
                "NA", # args.weights if 'mPU' in args.type else "NA",
                args.eta if args.eta != 0 else "NA",
                args.pert,
                args.trail
            ) for k in list(args.tag2Idx.keys())[1:] # discard 'O'
        ]
        trainer.performance_on_dataset(testSet, dp, args, "test", inference=True)

    # create dataset files with probabilities for bPUbN or confidence-based mPU
    elif args.add_probs:
        # manually pass the model path
        model_path = args.model_path
        testSet = dp.load_testset(args.dataset, "train.ALL.txt", args.no_lexicon)
        trainer.add_probs(dp, testSet, model_path, args.dataset, args.flag, args.added_suffix)

    # normal training
    else:
        trainSet, validSet, testSet, true_priors = dp.load_dataset(args.dataset, args.flag, args.pert, args.suffix, args.no_lexicon)
        print("train set size: {}, valid set size: {}, test set size: {}".format(len(trainSet), len(validSet), len(testSet)))

        trainer.train(trainSet, validSet, dp, args)

        model_dir = 'saved_model'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        model_path = os.path.join(model_dir, args.model_name)
        pred_path = os.path.join('predicted_data', args.dataset)
        if not os.path.exists(pred_path):
            os.mkdir(pred_path)

        for split_desc, predSet in [
            ("valid", validSet),
            ("test", testSet)
        ]:
            pred_file = pred_path + '/pred_' + f'{split_desc}_' + args.model_name + '.txt'
            try:
                overall = trainer.performance_on_testset(predSet, dp, args, split_desc, model_path, pred_file)
                print("\nOVERALL ON TEST: {}".format(overall))
            except FileNotFoundError:
                print('FileNotFound')

        print("\nArgs --> ", args.model_name)
        print("\n============DONE============\n\n\n")


if __name__ == "__main__":
    main()
