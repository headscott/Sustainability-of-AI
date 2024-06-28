from robustbench.data import load_cifar10
from robustbench.utils import load_model
from codecarbon import EmissionsTracker
import foolbox as fb
import numpy as np


def run_attack():
    emissions_list = []

    x_test, y_test = load_cifar10(n_examples=200)
    model = load_model(model_name='Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf')

    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    for i in range(1, 11):
        print(f"Schritt {i}: ")
        tracker = EmissionsTracker()
        tracker.start()
        _, advs, success = fb.attacks.LinfPGD()(fmodel, x_test.to('cuda:0'), y_test.to('cuda:0'), epsilons=[8 / 255])
        print('Robust accuracy {:.1%}'.format(1 - success.float().mean()))
        emissions: float = tracker.stop()
        emissions_list.append(emissions)

    # Mittelwert und Standardabweichung
    mean_emissions = np.mean(emissions_list)
    std_emissions = np.std(emissions_list)

    print(f"Mean emissions: {mean_emissions} kgCO2 equivalent")
    print(f"Standard deviation of emissions: {std_emissions} kgCO2 equivalent")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Start main run...")
    run_attack()
