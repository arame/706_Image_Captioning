import torch as T
import torch.nn as nn
import time
from average_meter import AverageMeter
from config import Hyper, Constants
from nltk.translate.bleu_score import corpus_bleu

def validate(val_loader, encoder, decoder, criterion):
    decoder.eval()  # eval mode (no dropout or batchnorm)
    encoder.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    with T.no_grad():
        # Batches
        i = 0
        for _, (imgs, captions) in enumerate(val_loader):
            i += 1
            # Move to device, if available
            imgs = imgs.to(Constants.device)
            captions = captions.to(Constants.device)

            # Forward prop.
            features = encoder(imgs)
            outputs = decoder(features, captions)
            vocab_size = outputs.shape[2]
            outputs1 = outputs.reshape(-1, vocab_size)
            captions1 = captions.reshape(-1)
            loss = criterion(outputs1, captions1)

            # Keep track of metrics
            losses.update(loss.item(), len(captions1))
            top5 = accuracy(outputs1, captions1, 5)
            top5accs.update(top5, len(captions1))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % Hyper.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image


        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(f'\n * LOSS - {losses.avg}, TOP-5 ACCURACY - {top5accs.avg}, BLEU-4 - {bleu4}\n')

    return bleu4

def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
