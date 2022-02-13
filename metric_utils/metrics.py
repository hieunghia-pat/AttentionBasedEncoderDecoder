import fastwer

class Metrics(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def get_error(self, label, true_label, mode):
        if mode == "character":
            return fastwer.score_sent(label,true_label, char_level=True)
        else:
            return fastwer.score_sent(label, true_label)

    def get_scores(self, predicteds, gts):
        cer = 0
        wer = 0
        batch_size = len(gts)
        for predicted, gt in zip(predicteds, gts):
            cer += self.get_error(predicted, gt, mode="character")
            wer += self.get_error(predicted, gt, mode="word")

        return {
            "cer": cer / batch_size,
            "wer": wer / batch_size
        }