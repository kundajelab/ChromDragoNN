import torch
from utils.data_utils import *
from utils import data_iterator
import torch.optim as optim
import torch.nn as nn
import os


def instantiate_model_stage1(args, Stage1Net, pipeline):
    model = Stage1Net(args)

    if args.resume_from_best:
        pipeline.load_checkpoint(model, checkpoint = args.checkpoint)
        print('LOADED WEIGHTS FROM ' + args.checkpoint + 'model_best.pth.tar')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model.to(device)
    
    return model


def instantiate_model_stage2(args, Stage2Net, pipeline):
    Stage1Net = import_net_from_file(args.stage1_file) 

    # load basset model
    basset_checkpoint = torch.load(args.stage1_pretrained_model_path + 'model_best.pth.tar')
    basset_args = basset_checkpoint['args']
    basset_model = Stage1Net(basset_args)
 
    BASSET_NUM_CELL_TYPES = basset_args.num_total_cell_types - len(basset_args.validation_list) - len(
        basset_args.test_list)
    
    # init model
    model = Stage2Net(BASSET_NUM_CELL_TYPES, basset_model, args)

    if args.resume_from_best:
        pipeline.load_checkpoint(model, checkpoint = args.checkpoint)
        print('LOADED WEIGHTS FROM ' + args.checkpoint + 'model_best.pth.tar')
    else:
        # load pretrained weights
        model.basset_model.load_state_dict(basset_checkpoint['state_dict'])
        print('LOADED STAGE 1 WEIGHTS FROM ' + args.stage1_pretrained_model_path + 'model_best.pth.tar')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model.to(device)
    
    return model

def load_data_iterator_stage1(args):
    return data_iterator.DataIterator(args.dnase, args.rna_quants, hold_out=args.hold_out,
                                    validation_list=args.validation_list, test_list=args.test_list, chromosomes=args.chromosomes)

def load_data_iterator_stage2(args):
    return data_iterator.DataIterator(args.dnase, args.rna_quants, hold_out=args.hold_out,
                                    validation_list=args.validation_list, test_list=args.test_list,
                                    balance_classes_train=True, positive_proportion=args.positive_proportion,
                                    return_locus_mean=args.with_mean, eval_subsample=500, chromosomes=args.chromosomes)

def run_stage1(model, di, args, pipeline):
    state = {k: v for k, v in args.items()}
    if args.resume_from_best:
        cp = torch.load(os.path.join(args.checkpoint, 'model_best.pth.tar'))
        best_val_loss = cp['best_loss']
        best_acc = cp['best_acc']
    else:
        best_val_loss = INF
        best_acc = 0
    
    optimizer = optim.Adam(model.parameters(), lr =state['lr'])
    criterion = pipeline.basset_loss

    start_epoch = 0

    if not args.epochs:
        _ = pipeline.test(model, optimizer, 1, di, args, criterion)

    for epoch in range(start_epoch, args.epochs):
        pipeline.adjust_learning_rate(optimizer, epoch, args)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs,state['lr']))

        train_loss, train_acc = pipeline.train(model, optimizer, epoch, di, args, criterion)
        test_loss, test_acc = pipeline.test(model, optimizer, epoch, di, args, criterion)

        # save model
        # is_best = test_loss < best_val_loss
        is_best = test_acc > best_acc
        if is_best:
            print("NEW BEST MODEL loss:{}\t auprc: {}".format(test_loss, test_acc))

        best_acc = max(test_acc, best_acc)
        best_val_loss = min(test_loss, best_val_loss)

        pipeline.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'acc': test_acc,
            'loss': test_loss,
            'best_loss': best_val_loss,
            'best_acc': best_acc,
            'hold_out': args.hold_out,
            'args': args,
            'validation_list': args.validation_list,
            'test_list': args.test_list,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)


def run_stage2(model, di, args, pipeline, mask_loss = False):
    state = {k: v for k, v in args.items()}
    if args.resume_from_best:
        cp = torch.load(os.path.join(args.checkpoint,'model_best.pth.tar'))
        best_acc = cp['best_acc']
        best_val_loss = cp['best_loss']
    else:
        best_acc = 0
        best_val_loss = INF

    print(model)

    cross_entropy_weights = torch.FloatTensor([1.0, args.pos_class_weight])
    if args.cuda:
        cross_entropy_weights = cross_entropy_weights.cuda()

    criterion = nn.NLLLoss(weight=cross_entropy_weights)

    if mask_loss:
        mask_criterion = lambda x: state['mask_L1']*torch.sum(torch.abs(x))/torch.numel(x)


    optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=state['lr'])
    start_epoch = 0

    if args.evaluate:
        if args.eval_on_validation:
            print("Evaluating on validation set")
            pipeline.evaluate_model(model, args, di, labels_avail=args.labels_avail, type='validation', mode=args.eval_mode)
        
        else:
            print("Evaluating on test set")
            pipeline.evaluate_model(model, args, di, labels_avail=args.labels_avail, type='test', mode=args.eval_mode)
            
    elif  not args.epochs:
           _ = pipeline.test(model, optimizer, 1, di, args, criterion)
    else:

        for epoch in range(start_epoch, args.epochs):
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

            if mask_loss:
                train_loss, train_acc = pipeline.train(model, optimizer, epoch, di, args, loss_criterion = criterion, mask_criterion = mask_criterion)
                test_loss, test_acc = pipeline.test(model, optimizer, epoch, di, args, loss_criterion = criterion, mask_criterion = mask_criterion)
            else:
                train_loss, train_acc = pipeline.train(model, optimizer, epoch, di, args, criterion)
                test_loss, test_acc = pipeline.test(model, optimizer, epoch, di, args, criterion)

            # save model
            # is_best = test_loss < best_val_loss
            is_best = test_acc > best_acc
            if is_best:
                print("NEW BEST MODEL loss:{}\t auprc: {}".format(test_loss, test_acc))

            best_acc = max(test_acc, best_acc)
            best_val_loss = min(test_loss, best_val_loss)

            pipeline.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'best_loss': best_val_loss,
                'pos_class_weight': args.pos_class_weight,
                'args': args,
                'hold_out': args.hold_out,
                'validation_list': args.validation_list,
                'test_list': args.test_list,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

            pipeline.adjust_learning_rate(optimizer, epoch, args, state)
