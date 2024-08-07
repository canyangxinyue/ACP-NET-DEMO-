import numpy as np


class RecognitionMetric():
    def __init__(self):
        pass
    
    def measure(self, batch, pred_texts):

        gt_texts = batch['texts'][0]
        
        error_sentences = 0
        total_CER = 0
        
        for gt_text, pred_text in zip(gt_texts,pred_texts):
            distance=self.minStringDistance(gt_text,pred_text)
            
            if distance>0:
                error_sentences+=1
            total_CER+=distance/len(gt_text)
            
        return [{"error_sentences":error_sentences, "total_CER": total_CER, "sentences":len(pred_texts)}]

    def validate_measure(self, batch, output):
        return self.measure(batch, output)

    def evaluate_measure(self, batch, output):
        return self.measure(batch, output), np.linspace(0, batch['image'].shape[0]).tolist()

    def gather_measure(self, raw_metrics):
        raw_metrics = [image_metrics
                       for batch_metrics in raw_metrics
                       for image_metrics in batch_metrics]

        error_sentences=0
        total_CER=0
        sentences=0
        
        for metric in raw_metrics:
            error_sentences+=metric['error_sentences']
            total_CER+=metric['total_CER']
            sentences+=metric['sentences']
            

        return {
            'ACC': (sentences-error_sentences)/sentences,
            'CER': total_CER/sentences
        }

    # 计算最小编辑距离
    def minStringDistance(self, true_label, pred_label):
        n1 = len(true_label)
        n2 = len(pred_label)
        # dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
        dp = np.zeros((n1 + 1, n2 + 1), dtype=np.int32)
        # 第一行
        for j in range(1, n2 + 1):
            dp[0][j] = dp[0][j - 1] + 1
        # 第一列
        for i in range(1, n1 + 1):
            dp[i][0] = dp[i - 1][0] + 1

        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                if true_label[i - 1] == pred_label[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1] ) + 1
        return dp[-1][-1]


    # 准确率计算函数
    def calculate_accuracy(self, true_labels, pred_labels):
        '''
        CER = (Sub + Ins + Del) / (Ins + Del + N)
        CER = (Sub + Ins + Del) / N = MSD / N
        SER
        '''
        assert len(true_labels) == len(pred_labels)
        sentences_count = len(true_labels)

        error_sentences = 0
        total_CER = 0
        for true, pred in zip(true_labels, pred_labels):
            MSD = self.minStringDistance(true, pred)
            if (MSD != 0):
                error_sentences += 1

            total_CER += MSD / len(true)

        avg_CER = total_CER / sentences_count
        avg_SER = error_sentences / sentences_count
        return avg_CER, avg_SER
