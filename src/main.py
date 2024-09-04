from helper.utils import load_yaml, pkl_and_write
from helper.args import get_command_line_args, replace_args_with_dict_values
from helper.noisy import NoiseAda
import torch
from helper.train_utils import train, test, get_optimizer, seed_everything, s_train
from models.nn import get_model
import numpy as np
import time
import logging
import torch.nn.functional as F
from copy import deepcopy
from helper.data import get_dataset
import os.path as osp
import optuna
from helper.hyper_search import hyper_search
import sys
from tqdm import tqdm
import matplotlib
#matplotlib.use('Agg')
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def train_pipeline(seeds, args, epoch, data, need_train, need_save_logits, reliability_list):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_result_acc = []
    early_stop_accum = 0
    val_result_acc = []
    out_res = []
    best_val = 0
    debug_accs = []
    train_accs = []
    num_of_classes = data.y.max().item() + 1
    if args.model_name == 'S_model':
        noise_ada = NoiseAda(num_of_classes).to(device)
    else:
        noise_ada = None

    all_train_accuracies = []
    all_test_accuracies = []

    for i, seed in enumerate(seeds):
        if len(reliability_list) > 0:
            reliability = reliability_list[0].to(device)
        seed_everything(seed)
        model = get_model(args).to(device)
        optimizer, scheduler = get_optimizer(args, model)
        if args.loss_type == 'ce':
            loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)            
        else:
            loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction='none')            
        if args.normalize:
            data.x = F.normalize(data.x, dim = -1)
        data = data.to(device)
        data.train_mask = data.train_masks[i]
        data.val_mask = data.val_masks[i]
        data.test_mask = data.test_masks[i]
        debug_acc = []
        this_train_acc = []
        if 'ft' in args.data_format and 'no_ft' not in args.data_format:
            data.x = data.xs[i]
            data.train_mask = data.train_masks[i]
            data.val_mask = data.val_masks[i]
            data.test_mask = data.test_masks[i]
        if 'pl' in args.split or 'active' in args.split:
            data.train_mask = data.train_masks[i]
            data.val_mask = data.val_masks[i]
            data.test_mask = data.test_masks[i]
            data.backup_y = data.y.clone()
            if not args.debug_gt_label:
                data.y = data.ys[i]
            else:
                print("Using ground truth label")
        train_accuracies = []
        test_accuracies = []
        for j in tqdm(range(epoch)):
            train_mask = data.train_mask
            val_mask = data.val_mask
            if need_train:
                if 'rim' in args.strategy or 'iterative' in args.strategy or args.split == 'active_train':
                    train_loss, val_loss, val_acc, train_acc = train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val, reliability)
                else:
                    if args.model_name == 'S_model':
                        train_loss, val_loss, val_acc, train_acc = s_train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val, noise_ada)
                    else:
                        train_loss, val_loss, val_acc, train_acc = train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val)
                if scheduler:
                    scheduler.step()
                if args.output_intermediate and not args.no_val:
                    print(f"Epoch {j}: Train loss: {train_loss}, Val loss: {val_loss}, Val acc: {val_acc[0]}")
                if args.debug:
                    if args.filter_strategy == 'none':
                        test_acc, res = test(model, data, 0, data.test_mask)
                        #print("test-accuracy1",test_acc)
                    else:
                        test_acc, res = test(model, data, 0, data.test_mask, data.backup_y)
                        print("test-accuracy2",test_acc)
                    test_accuracies.append(test_acc)
                    
                    debug_acc.append(test_acc)
                    this_train_acc.append(train_acc)
                if not args.no_val:
                    if val_acc > best_val:
                        best_val = val_acc
                        best_model = deepcopy(model)
                        early_stop_accum = 0
                    else:
                        if j >= args.early_stop_start:
                            early_stop_accum += 1
                        if early_stop_accum > args.early_stopping and j >= args.early_stop_start:
                            print(f"Early stopping at epoch {j}")
                            break
            else:
                best_model = model
            if args.debug:
                train_accuracies.append(train_acc)
                test_acc, _ = test(model, data, args.return_embeds, data.test_mask)
                #print("test-accuracy3",test_acc)
                #test_accuracies.append(test_acc)
        if 'pl' in args.split or 'active' in args.split:
            data.y = data.backup_y
        if args.no_val or best_model is None:
            best_model = model
        test_acc, res = test(best_model, data, args.return_embeds, data.test_mask)
        #print("test-accuracy4",test_acc)
        test_result_acc.append(test_acc)
        val_result_acc.append(best_val)
        out_res.append(res)
        best_val = 0
        best_model = None
        if args.debug:
            debug_accs.append(debug_acc)
            train_accs.append(this_train_acc)
        if need_save_logits:
            torch.save(out_res, f'../output/logits/{args.dataset}_{args.split}_{args.model_name}_{seed}_logits.pt')

        all_train_accuracies.append(train_accuracies)
        all_test_accuracies.append(test_accuracies)

    if not args.debug:
        return test_result_acc, val_result_acc, out_res
    else:
        return test_result_acc, val_result_acc, out_res, debug_accs, train_accs, all_train_accuracies, all_test_accuracies


def plot_accuracies(all_train_accuracies, all_test_accuracies, strategies, dataset, split, model_name):
    print(f"Creating figure with size (14,7)")
    plt.figure(figsize=(14, 7))
    epochs = len(all_train_accuracies[0][0])
    x = np.arange(epochs)

    for i, strategy in enumerate(strategies):
        train_accuracies = all_train_accuracies[i]
        test_accuracies = all_test_accuracies[i]
        
        train_mean = np.mean(train_accuracies, axis=0)
        train_std = np.std(train_accuracies, axis=0)
        test_mean = np.mean(test_accuracies, axis=0)
        test_std = np.std(test_accuracies, axis=0)
        
        plt.plot(x, train_mean, label=f'{strategy} Train Acc', linestyle='-')
        plt.fill_between(x, train_mean - train_std, train_mean + train_std, alpha=0.2)
        
        plt.plot(x, test_mean, label=f'{strategy} Test Acc', linestyle='--')
        plt.fill_between(x, test_mean - test_std, test_mean + test_std, alpha=0.2)
    
    plt.xlabel("Epochs", fontsize=25, fontweight='bold', family='sans-serif')
    plt.ylabel("Accuracy", fontsize=25, fontweight='bold', family='sans-serif')
    plt.title(f"Training and Testing Accuracy for {model_name} on {dataset} ({split} split)", fontsize=25, fontweight='bold', family='sans-serif')
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dataset}_{split}_{model_name}_strategies_accuracies.png")
    plt.savefig(f"{dataset}_{split}_{model_name}_strategies_accuracies.pdf")
    plt.show()


def main(data_path, args=None, custom_args=None, save_best=False):
    seeds = [i for i in range(args.main_seed_num)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if custom_args is not None:
        args = replace_args_with_dict_values(args, custom_args)
    vars(args)['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    params_dict = load_yaml(args.yaml_path)
    logit_path = params_dict['LOGIT_PATH']

    reliability_list = []

    strategies = [
        "cited_cites",
        "cited",
        "cites",
        "no_neighbors",
        #"pred_pred",
        #"pred_succ",
        #"succ_pred",
        #"succ_succ"
        # Add more labels as needed
    ]

    all_train_accuracies = []
    all_test_accuracies = []

    for strategy in strategies:
        vars(args)['filter_strategy'] = strategy
        data = get_dataset(seeds, args.dataset, args.split, args.data_format, data_path, logit_path, args.pl_noise, args.no_val, args.budget, args.strategy, args.num_centers, args.compensation, args.save_data, args.filter_strategy, args.max_part, args.oracle, reliability_list, args.total_budget, args.second_filter, True, False, args.filter_all_wrong_labels, args.alpha, args.beta, args.gamma, args.ratio).to(device)
        epoch = args.epochs

        #get_dataset(seeds, dataset, split, data_format, data_path, logit_path, random_noise = 0, no_val = 1, budget = 20, strategy = 'random', num_centers = 0, compensation = 0, save_data = 0, llm_strategy = 'none', max_part = 0, oracle_acc = 1, reliability_list = None, total_budget = -1, second_filter = 'none', train_stage = True, post_pro = False, filter_all_wrong_labels = False, alpha = 0.1, beta = 0.1, gamma = 0.1, ratio = 0.3, fixed_data = True):


        vars(args)['input_dim'] = data.x.shape[1]
        vars(args)['num_classes'] = data.y.max().item() + 1

        if args.model_name == 'LP':
            need_train = False
        else:
            need_train = True

        if not args.batchify and args.ensemble_string == "":
            data.x = data.x.to(torch.float32)
            if not args.debug:
                test_result_acc, _, _ = train_pipeline(seeds, args, epoch, data, need_train, args.save_logits, reliability_list)
            else:
                test_result_acc, _, _, debug_accs, train_accs, train_accuracies, test_accuracies = train_pipeline(seeds, args, epoch, data, need_train, args.save_logits, reliability_list)
                all_train_accuracies.append(train_accuracies)
                all_test_accuracies.append(test_accuracies)

            mean_test_acc = np.mean(test_result_acc)
            std_test_acc = np.std(test_result_acc)

            if args.debug:
                best_possible_test_acc = [max(res) for res in debug_accs]
                res_train_accs = [x[-1] for x in train_accs]
                print(f"Train Accuracy: {np.mean(res_train_accs) * 100:.2f} ± {np.std(res_train_accs) * 100:.2f}")
                print(f"Test Accuracy: {mean_test_acc:.2f} ± {std_test_acc:.2f}")
                print(f"Best possible accuracy: {np.mean(best_possible_test_acc) * 100:.2f} ± {np.std(best_possible_test_acc) * 100:.2f}")
                print("Test acc: {}".format(test_result_acc))

            if save_best:
                pkl_and_write(args, osp.join("./bestargs", f"{args.model_name}_{args.dataset}_{args.data_format}.pkl"))

            if args.debug:
                if args.debug_gt_label:
                    pkl_and_write(debug_accs, osp.join("./debug", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}_gt.pkl"))
                    pkl_and_write(train_accs, osp.join("./debug_train", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}_train_accs_gt.pkl"))
                elif args.filter_all_wrong_labels:
                    pkl_and_write(debug_accs, osp.join("./debug", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}_filtered.pkl"))
                    pkl_and_write(train_accs, osp.join("./debug_train", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}_train_accs_filtered.pkl"))
                else:
                    pkl_and_write(debug_accs, osp.join("./debug", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}.pkl"))
                    pkl_and_write(train_accs, osp.join("./debug_train", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}_train_accs.pkl"))

    if args.debug:
        plot_accuracies(all_train_accuracies, all_test_accuracies, strategies, args.dataset, args.split, args.model_name)


if __name__ == '__main__':
    current_time = int(time.time())
    print("Start")

    args = get_command_line_args()    
    params_dict = load_yaml(args.yaml_path)
    data_path = params_dict['DATA_PATH']
    if args.mode == "main":
        main(data_path, args=args)
