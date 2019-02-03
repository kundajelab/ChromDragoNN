import torch
from utils.data_utils import *
import torch.optim as optim
import torch.nn as nn
import os


def run_stage1(model, di, args, pipeline):
    state = {k: v for k, v in args.items()}
    if args.resume_from_best:
        cp = torch.load(os.path.join(args.checkpoint, 'model_best.pth.tar'))
        best_val_loss = cp['best_loss']
        best_acc = cp['best_acc']
    else:
        best_val_loss = INF
        best_acc = 0
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model.to(device)
    #if args.cuda:
    #    model.cuda()
    #else:
    #    model.cpu()

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
            'state_dict': model.state_dict(),
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model.to(device)
    #if args.cuda:
    #    model.cuda()
    #else:
    #    model.cpu()

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
                'state_dict': model.state_dict(),
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
