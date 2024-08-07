import numpy as np
import cv2
from functools import reduce
import string
import torch

def get_vocabulary(voc_type, EOS='EOS', PADDING='PAD', UNKNOWN='UNK'):
    if voc_type == 'LOWERCASE':
        voc = list(string.digits + string.ascii_lowercase)
    elif voc_type == 'ALLCASES':
        voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'ALLCASES_SYMBOLS':
        voc = list(string.printable[:-6])
    else:
        raise KeyError('voc_type must be one of "LOWERCASE", '
                       '"ALLCASES", "ALLCASES_SYMBOLS"')

    # update the voc with specifical chars
    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    char2id = dict(zip(voc, range(len(voc))))
    id2char = dict(zip(range(len(voc)), voc))

    return voc, char2id, id2char

class MakewordMap():
    r'''
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''

    def __init__(self, max_word_num=200, max_word_len=32, shrink_type='pyclipper'):
        self.voc, self.char2id, self.id2char = get_vocabulary('LOWERCASE')
        self.max_word_num = max_word_num
        self.max_word_len = max_word_len

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        img = data['img']
        bboxes = data['text_polys']

        words= data['words']

        if bboxes.shape[0] > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]
            words = words[:self.max_word_num]

        gt_words = np.full((self.max_word_num + 1, self.max_word_len),
                           self.char2id['PAD'],
                           dtype=np.int32)
        word_mask = np.zeros((self.max_word_num + 1,), dtype=np.int32)
        for i, word in enumerate(words):
            if word == '###':
                continue
            word = word.lower()
            gt_word = np.full((self.max_word_len,),
                              self.char2id['PAD'],
                              dtype=np.int)
            for j, char in enumerate(word):
                if j > self.max_word_len - 1:
                    break
                if char in self.char2id:
                    gt_word[j] = self.char2id[char]
                else:
                    gt_word[j] = self.char2id['UNK']
            if len(word) > self.max_word_len - 1:
                gt_word[-1] = self.char2id['EOS']
            else:
                gt_word[len(word)] = self.char2id['EOS']
            gt_words[i + 1] = gt_word
            word_mask[i + 1] = 1

        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        if len(bboxes) > 0:
            for i in range(len(bboxes)):
                bbox=bboxes[i].astype('int32')

                cv2.drawContours(gt_instance, [bbox], -1, i + 1, -1)
                if words[i] == '###':
                    cv2.drawContours(training_mask, [bbox], -1, 0, -1)


        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1


        max_instance = np.max(gt_instance)
        gt_bboxes = np.zeros((self.max_word_num + 1, 4), dtype=np.int32)
        for i in range(1, max_instance + 1):
            ind = gt_instance == i
            if np.sum(ind) == 0:
                continue
            points = np.array(np.where(ind)).transpose((1, 0))
            tl = np.min(points, axis=0)
            br = np.max(points, axis=0) + 1
            gt_bboxes[i] = (tl[0], tl[1], br[0], br[1])

        data['gt_texts'] = gt_text
        data['gt_bboxes'] = gt_bboxes
        data['gt_words'] =  torch.from_numpy(gt_words).long()
        data['word_mask'] = word_mask
        data['gt_instance'] = gt_instance
        data['training_mask'] = training_mask

        return data


if __name__ == '__main__':
    from shapely.geometry import Polygon
    import pyclipper


