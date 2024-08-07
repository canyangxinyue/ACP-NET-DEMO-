import numpy as np

from .detection.iou import DetectionIoUEvaluator
from ...util import ctc_greedy_decoder
from ...util import minStringDistance
import copy

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self


class E2EQuadMetric():
    def __init__(self, is_output_polygon=False):
        self.is_output_polygon = is_output_polygon
        self.evaluator = DetectionIoUEvaluator(is_output_polygon=is_output_polygon)

    def measure(self, batch, output, box_thresh=0.6):
        '''
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        output: (polygons, ...)
        '''
        results = []
        gt_polyons_batch = batch['text_polys']
        ignore_tags_batch = batch['ignore_tags']
        pred_polygons_batch = np.array(output[0])
        pred_scores_batch = np.array(output[1])
        for polygons, pred_polygons, pred_scores, ignore_tags in zip(gt_polyons_batch, pred_polygons_batch, pred_scores_batch, ignore_tags_batch):
            gt = [dict(points=np.int64(polygons[i]), ignore=ignore_tags[i]) for i in range(len(polygons))]
            if self.is_output_polygon:
                pred = [dict(points=pred_polygons[i]) for i in range(len(pred_polygons))]
            else:
                pred = []
                # print(pred_polygons.shape)
                for i in range(pred_polygons.shape[0]):
                    if pred_scores[i] >= box_thresh:
                        # print(pred_polygons[i,:,:].tolist())
                        pred.append(dict(points=pred_polygons[i, :, :].astype(np.int)))
                # pred = [dict(points=pred_polygons[i,:,:].tolist()) if pred_scores[i] >= box_thresh for i in range(pred_polygons.shape[0])]
            results.append(self.evaluator.evaluate_image(gt, pred))
        return results

    def validate_measure(self, batch, output, box_thresh=0.6):
        return self.measure(batch, output, box_thresh)

    def evaluate_measure(self, batch, output):
        return self.measure(batch, output), np.linspace(0, batch['image'].shape[0]).tolist()

    def gather_measure(self, raw_metrics):
        raw_metrics = [image_metrics
                       for batch_metrics in raw_metrics
                       for image_metrics in batch_metrics]

        result = self.evaluator.combine_results(raw_metrics)

        precision = AverageMeter()
        recall = AverageMeter()
        fmeasure = AverageMeter()

        precision.update(result['precision'], n=len(raw_metrics))
        recall.update(result['recall'], n=len(raw_metrics))
        fmeasure_score = 2 * precision.val * recall.val / (precision.val + recall.val + 1e-8)
        fmeasure.update(fmeasure_score)

        return {
            'precision': precision,
            'recall': recall,
            'fmeasure': fmeasure
        }

    # rec_labels_batches：[ [box区域, 识别输出tensor，score]    *boxes个]  *batch个
    def measure_with_rec(self,raw_metrics,batch,rec_labels_b ,label2char:dict):
        gt_text_batches=batch['texts']
        raw_metrics_e2e= copy.deepcopy(raw_metrics)
        for gt_textes, rec_labels, raw_metric in zip(gt_text_batches,rec_labels_b, raw_metrics_e2e):
            new_pairs=[]
            matched=0
            for pair in raw_metric['pairs']:
                pred_texts=("".join([label2char[label] for label in ctc_greedy_decoder(rec_labels[pair['det']])[0]]))
                gt=gt_textes[pair['gt']]
                if pred_texts==gt:
                    new_pairs.append(pair)
                    matched+=1
            raw_metric['pairs']=new_pairs
            raw_metric['detMatched']=matched
        
        return raw_metrics_e2e
    
    def gather_rec_measure(self, metrics):
        sentencesList, errSentencesList, total_CERList=[],[],[]
        for metric in metrics:
            sentencesList.append(metric[0])
            errSentencesList.append(metric[1])
            total_CERList.append(metric[2])
        total_sentences = sum(sentencesList)
        error_sentences = sum(errSentencesList)
        total_CER = sum(total_CERList)
        return {
            'CER': total_CER/ total_sentences,
            'SER': error_sentences/ total_sentences
        }
                
                