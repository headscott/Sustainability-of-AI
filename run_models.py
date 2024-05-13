from robustbench.data import load_cifar10
from robustbench.utils import load_model
from codecarbon import EmissionsTracker
import foolbox as fb


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tracker = EmissionsTracker()
    tracker.start()
    try:
        x_test, y_test = load_cifar10(n_examples=50)
        model = load_model(model_name='Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf')

        fmodel = fb.PyTorchModel(model, bounds=(0, 1))

        _, advs, success = fb.attacks.LinfPGD()(fmodel, x_test.to('cuda:0'), y_test.to('cuda:0'), epsilons=[8 / 255])
        print('Robust accuracy: {:.1%}'.format(1 - success.float().mean()))
    finally:
        tracker.stop()