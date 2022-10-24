import torch

import penne


###############################################################################
# Constants
###############################################################################


# Evaluation threshold for RPA and RCA
THRESHOLD = 50  # cents


###############################################################################
# Aggregate metric
###############################################################################


class Metrics:

    def __init__(self):
        self.accuracy = Accuracy()
        self.f1 = F1()
        self.loss = Loss()
        self.rca = RCA()
        self.rmse = RMSE()
        self.rpa = RPA()

    def __call__(self):
        return (
            self.accuracy() |
            self.f1() |
            self.loss() |
            self.rca() |
            self.rmse() |
            self.rpa())

    def update(self, logits, bins, target, voiced):
        # Detach from graph
        logits = logits.detach().cpu()
        bins = bins.cpu()

        # Update loss
        self.loss.update(logits, bins)

        # Decode bins, pitch, and periodicity
        predicted = logits.argmax(dim=1, keepdims=True)
        pitch = penne.convert.bins_to_frequency(predicted)
        periodicity = logits.max(dim=1, keepdims=True).values

        # Mask unvoiced
        pitch, target = pitch[voiced], target[voiced]

        # Update pitch metrics
        self.accuracy.update(predicted[voiced], bins[voiced])
        self.rca.update(pitch, target)
        self.rmse.update(pitch, target)
        self.rpa.update(pitch, target)

        # Update periodicity metrics
        self.f1.update(periodicity, voiced)

    def reset(self):
        self.accuracy.reset()
        self.f1.reset()
        self.loss.reset()
        self.rca.reset()
        self.rmse.reset()
        self.rpa.reset()


###############################################################################
# Individual metrics
###############################################################################


class Accuracy:

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'accuracy': (self.true_positives / self.count).item()}

    def update(self, predicted, target):
        self.true_positives += (predicted == target).sum()
        self.count += predicted.shape[-1]

    def reset(self):
        self.true_positives = 0
        self.count = 0


class F1:

    def __init__(self, thresholds=None):
        self.thresholds = \
            [2 ** -i for i in range(1, 9)] if thresholds is None \
            else thresholds
        self.precision = [Precision() for _ in range(len(self.thresholds))]
        self.recall = [Recall() for _ in range(len(self.thresholds))]

    def __call__(self):
        result = {}
        iterator = zip(self.thresholds, self.precision, self.recall)
        for threshold, precision, recall in iterator:
            precision = precision()['precision']
            recall = recall()['recall']
            f1 = 2 * precision * recall / (precision + recall)
            result |= {
                f'f1-{threshold:.6f}': f1,
                f'precision-{threshold:.6f}': precision,
                f'recall-{threshold:.6f}': recall}
        return result

    def update(self, periodicity, voiced):
        iterator = zip(self.thresholds, self.precision, self.recall)
        for threshold, precision, recall in iterator:
            predicted = periodicity > threshold
            precision.update(predicted, voiced)
            recall.update(predicted, voiced)

    def reset(self):
        """Reset the F1 score"""
        for precision, recall in zip(self.precision, self.recall):
            precision.reset()
            recall.reset()


class Loss():

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'loss': (self.total / self.count).item()}

    def update(self, logits, bins):
        self.total += penne.train.loss(logits, bins)
        self.count += bins.shape[-1]

    def reset(self):
        self.count = 0
        self.total = 0.


class Precision:

    def __init__(self):
        self.reset()

    def __call__(self):
        precision = (
            self.true_positives /
            (self.true_positives + self.false_positives)).item()
        return {'precision': precision}

    def update(self, predicted, voiced):
        self.true_positives += (predicted & voiced).sum()
        self.false_positives += (predicted & ~voiced).sum()

    def reset(self):
        self.true_positives = 0
        self.false_positives = 0


class Recall:

    def __init__(self):
        self.reset()

    def __call__(self):
        recall = (
            self.true_positives /
            (self.true_positives + self.false_negatives)).item()
        return {'recall': recall}

    def update(self, predicted, voiced):
        self.true_positives += (predicted & voiced).sum()
        self.false_negatives += (~predicted & voiced).sum()

    def reset(self):
        self.true_positives = 0
        self.false_negatives = 0


class RCA:
    """Raw chroma accuracy"""

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'rca': (self.sum / self.count).item()}

    def update(self, predicted, target):
        # Compute pitch difference in cents
        difference = cents(predicted, target)

        # Forgive octave errors
        difference[difference > (penne.OCTAVE - THRESHOLD)] -= penne.OCTAVE
        difference[difference < -(penne.OCTAVE - THRESHOLD)] += penne.OCTAVE

        # Count predictions that are within 50 cents of target
        self.sum += (torch.abs(difference) < THRESHOLD).sum()
        self.count += predicted.shape[-1]

    def reset(self):
        self.count = 0
        self.sum = 0


class RMSE:
    """Root mean square error of pitch distance in cents"""

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'rmse': (torch.sqrt(self.sum / self.count)).item()}

    def update(self, predicted, target):
        self.sum += (cents(predicted, target) ** 2).sum()
        self.count += predicted.shape[-1]

    def reset(self):
        """Reset the WRMSE score"""
        self.count = 0
        self.sum = 0


class RPA:
    """Raw prediction accuracy"""

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'rpa': (self.sum / self.count).item()}

    def update(self, predicted, target):
        difference = cents(predicted, target)
        self.sum += (torch.abs(difference) < THRESHOLD).sum()
        self.count += predicted.shape[-1]

    def reset(self):
        self.count = 0
        self.sum = 0


###############################################################################
# Individual metrics
###############################################################################


def cents(a, b):
    """Compute pitch difference in cents"""
    return penne.OCTAVE * torch.log2(a / b)