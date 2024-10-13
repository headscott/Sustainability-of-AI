import os
import torch
import foolbox as fb
from autoattack import AutoAttack
from robustbench.data import load_cifar100
from robustbench.utils import clean_accuracy
from robustbench.utils import load_model
from codecarbon import EmissionsTracker
from glob import glob
import time


def run_all(param_config):
    # Liste aller Modelle mit CIFAR-100 und Linf epsilon=8/255
    # bei Bai2024MixedNUTS kam es immer zu einem Fehler:
    # [Errno 2] No such file or directory: 'path/to/lib/python3.12/site-packages/robustbench/model_zoo/architectures/optimal_spca.yaml'
    labels_cifar100_linf = [
        'Addepalli2021Towards_PARN18',
        'Addepalli2021Towards_WRN34',
        'Addepalli2022Efficient_RN18',
        'Addepalli2022Efficient_WRN_34_10',
        'Bai2023Improving_edm',
        'Bai2023Improving_trades',
        'Bai2024MixedNUTS',
        'Chen2020Efficient',
        'Chen2021LTD_WRN34_10',
        'Chen2024Data_WRN_34_10',
        'Cui2020Learnable_34_10_LBGAT0',
        'Cui2020Learnable_34_10_LBGAT6',
        'Cui2020Learnable_34_10_LBGAT9_eps_8_255',
        'Cui2020Learnable_34_20_LBGAT6',
        'Cui2023Decoupled_WRN-28-10',
        'Cui2023Decoupled_WRN-34-10',
        'Cui2023Decoupled_WRN-34-10_autoaug',
        'Debenedetti2022Light_XCiT-L12',
        'Debenedetti2022Light_XCiT-M12',
        'Debenedetti2022Light_XCiT-S12',
        'Gowal2020Uncovering',
        'Gowal2020Uncovering_extra',
        'Hendrycks2019Using',
        'Jia2022LAS-AT_34_10',
        'Jia2022LAS-AT_34_20',
        'Pang2022Robustness_WRN28_10',
        'Pang2022Robustness_WRN70_16',
        'Rade2021Helper_R18_ddpm',
        'Rebuffi2021Fixing_28_10_cutmix_ddpm',
        'Rebuffi2021Fixing_70_16_cutmix_ddpm',
        'Rebuffi2021Fixing_R18_ddpm',
        'Rice2020Overfitting',
        'Sehwag2021Proxy',
        'Sitawarin2020Improving',
        'Wang2023Better_WRN-28-10',
        'Wang2023Better_WRN-70-16',
        'Wu2020Adversarial'
    ]

    if param_config == 1:
        for label in labels_cifar100_linf:
            run_clean_auto(label)
    elif param_config == 2:
        for label in labels_cifar100_linf:
            run_inference(label)
    elif param_config == 3:
        for label in labels_cifar100_linf:
            run_one_inference(label)
    elif param_config == 4:
        for label in labels_cifar100_linf:
            get_complexity(label)


# Mit dieser Methode habe ich die AutoAttack und Clean Accuracy getestet
def run_clean_auto(label):
    print(f"Versuche {label} zu evaluieren...")
    tracker = EmissionsTracker(project_name=label, experiment_name="Inferenz_Clean_Auto_" + label[:15])

    x_test, y_test = load_cifar100(n_examples=100)
    device = torch.device("cuda:0")
    x_test = torch.tensor(x_test).to(device)
    y_test = torch.tensor(y_test).to(device)
    try:
        model = load_model(model_name=label, dataset='cifar100', threat_model='Linf')

        fmodel = fb.PyTorchModel(model, bounds=(0, 1))

        # init AutoAttack
        adversary = AutoAttack(fmodel, norm='Linf', eps=8 / 255, version='standard')

        for i in range(1, 6):
            print(f"Schritt {i}: ")

            # Clean Accuracy:
            tracker.start_task(f"Clean_{i}", )
            acc = clean_accuracy(model, x_test, y_test)
            tracker.stop_task()
            print(f'CIFAR-100 Clean Accuracy: {acc: .1%}')

            # AutoAttack:
            tracker.start_task(f"AutoAttack_{i}", )
            advs = adversary.run_standard_evaluation(x_test, y_test, bs=50)
            predictions_adv = fmodel(advs).argmax(dim=-1)
            auto_accuracy = (predictions_adv == y_test).float().mean().item()
            tracker.stop_task()
            print(f'CIFAR-100 AutoAttack Accuracy: {auto_accuracy * 100: .1f}%')
    except Exception as error:
        print("Ein Fehler bei: " + label + "\n", error)
    finally:
        tracker.stop()


# Mit dieser Methode habe ich die Inferenz auf 1000 Bildern aus dem CIFAR-100 Datensatz mit CodeCarbon durchgeführt
def run_inference(label):
    print(f"Versuche {label} Inferenzzeit zu messen...")
    tracker = EmissionsTracker(project_name=label, experiment_name="Inferenzzeit_single_" + label[:15])

    x_test, y_test = load_cifar100(n_examples=1000)  # y_test kann hier ignoriert werden (wahres Label)
    device = torch.device("cuda:0")
    x_test = torch.tensor(x_test).to(device)
    try:
        # Laden des Modells von RobustBench
        model = load_model(model_name=label, dataset='cifar100', threat_model='Linf')

        # Auf GPU 0
        model = model.to(device)

        for i in range(1, 50):
            print(f"Schritt {i}...")

            # Inferenzzeit:
            tracker.start_task(f"Inferenzzeit_single_{i}", )
            with torch.no_grad():
                model(x_test)
            tracker.stop_task()

    except Exception as error:
        print("Ein Fehler bei: " + label + "\n", error)
    finally:
        tracker.stop()


#  Mit dieser Methode habe ich die Inferenz auf einem Bildern aus dem CIFAR-100 Datensatz ohne CodeCarbon durchgeführt
def run_one_inference(label):
    print(f"Versuche {label} Inferenzzeit zu messen...")

    x_test, y_test = load_cifar100(n_examples=1)  # y_test kann hier ignoriert werden (wahres Label)
    device = torch.device("cuda:0")
    x_test = torch.tensor(x_test).to(device)
    try:
        model = load_model(model_name=label, dataset='cifar100', threat_model='Linf')

        model = model.to(device)

        for i in range(1, 101):
            # CUDA synchronisieren, um genaue Messungen zu gewährleisten
            torch.cuda.synchronize()  # Synchronisiere vor der Messung
            start_time = time.time()

            with torch.no_grad():
                model(x_test)

            torch.cuda.synchronize()  # Synchronisiere nach der Ausführung
            inference_time = time.time() - start_time

            print(f"Inferenzzeit für Schritt {i}: {inference_time:.4f} Sekunden")

    except Exception as error:
        print("Ein Fehler bei: " + label + "\n", error)


# Bestimmt die maximale Tiefe des gesamten Modells (Hilfsfunktion)
def get_model_depth(model, current_depth=0):
    # Modell hat keine weiteren Kinder -> Blatt, also ist Tiefe current_depth + 1
    if len(list(model.children())) == 0:
        return current_depth + 1
    # Modell hat Kinder -> rekursiv die Tiefe jedes Kindes bestimmen
    else:
        return max([get_model_depth(child, current_depth + 1) for child in model.children()])


# Bestimmt Eigenschaften wie die Anzahl der Parameter, die Tiefe des Modells und die Groesse auf dem Datentraeger
def get_complexity(label):
    try:
        # Laden des Modells von RobustBench
        model = load_model(model_name=label, dataset='cifar100', threat_model='Linf')

        # Berechnen der Anzahl der Parameter
        num_params = sum(p.numel() for p in model.parameters())

        # Pfad zum Ordner mit allen Modell Dateien
        model_folder = "./models/cifar100/Linf/"

        # Finde Modell entsprechend des angegebenen Labels in Modell Ordner
        model_files = glob(f"{model_folder}{label}.*")

        if model_files:
            # Da nur ein Modell existiert, nimm das erste
            model_path = model_files[0]
            # Bestimme Dateigroesse in Megabyte
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

            # Bestimme Tiefe des Modells
            depth = get_model_depth(model)

            print(f'Number Params for {label}: {num_params} and Model Size on Disk: {model_size_mb:.2f} MB, Tiefe: {depth}')
        else:
            # Falls Modell nicht im Modell Ordner exisitert, erstelle temporaere Datei
            tmp_model = f"./{label}_tmp_model.pth"
            torch.save(model.state_dict(), tmp_model)

            # Bestimme Dateigroesse der temporaeren Datei in Megabyte
            model_size_mb = os.path.getsize(tmp_model) / (1024 * 1024)
            
            # Bestimme Tiefe des Modells
            depth = get_model_depth(model)

            print(f"Number Params for {label}: {num_params} and Model Size on Disk: {model_size_mb:.2f} MB (temporarly saved), Tiefe: {depth}")

            # Entferne temporaere Modell Datei
            os.remove(tmp_model)
    except Exception as error:
        print("Ein Fehler bei: " + label + "\n", error)


if __name__ == '__main__':
    run_all(1)  # fuerht den test mit AutoAttack und Clean Accuracy aus
    run_all(2)  # fuerht den test mit 1000 Bildern aus CIFAR-100 und CodeCarbon aus
    run_all(3)  # fuerht den test mit einem Bild aus CIFAR-100 ohne CodeCarbon (nur Laufzeit)
    run_all(4)  # gibt Tiefe, Anzahl Parameter und Disk Size der jeweiligen Modelle aus (, die im Unterordner /models/cifar100/Linf/ liegen sollten)
