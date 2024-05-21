import torch
from torch import nn
import torch.nn.functional as F
from data import ClassificationTwoSpiralsData
from visualization_utils import inspect_data, plot_data, x_data_from_grid
import matplotlib.pyplot as plt


class TorchMultiLayerNetwork(nn.Module):
    """
    Prosta sieć neuronowa w pyTorch o parametryzowalnej liczbie warstw ukrytych.
    Konstrukcja analogiczna, jak dwuwarstwowej sieci z części pierwszej.

    Np. dla n_in=2, n_hiddens=[10, 20] i n_out=1 powstanie sieć z dwoma wejściami, pierwszą warstwą ukrytą z 10
    neuronami, drugą warstwą ukrytą z 20 neuronami i jednym neuronem w warstwie wyjściowej. Funkcje aktywacji
    ustawione "na sztywno": 'relu' w warstwach ukrytych i 'sigmoid' w warstwie wyjściowej.
    """

    def __init__(self, n_in, n_hiddens, n_out):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(n_i, n_o) for n_i, n_o in zip([n_in] + n_hiddens, n_hiddens + [n_out])])

    def forward(self, x_data):
        x = x_data
        for l in self.layers[:-1]:
            x = F.relu(l(x))
        return F.sigmoid(self.layers[-1](x))


def evaluate_model(model, x, y):
    with torch.no_grad():                       # przy ewaluacji nie ma potrzeby liczyć gradientów
        y_pred = torch.round(model.forward(x))
        acc = (y == y_pred).sum().item() / y.shape[0]
    return acc


def step_learning_rate_schedule (initial_lr, step_size, gamma, current_epoch):
    """
    Step-based learning rate schedule.

    Parameters:
    initial_lr (float): The initial learning rate (the learning rate at epoch 0).
    step_size (int): The number of epochs between learning rate reductions.
    gamma (float): The factor by which the learning rate should be reduced at each step.
    current_epoch (int): The current epoch number.

    Returns:
    float: The learning rate for the current epoch.
    """
    lr = initial_lr * (gamma ** (current_epoch // step_size))
    return lr


def training(model, x, y):

    # training hiperparameters
    n_steps = 50000
    learning_rate = 0.35        # try different values

    initial_learning_rate = 0.35
    step_size = 2000
    gamma = 0.992

    # 0.4 1000 0.992   train: 94.25%    test: 94.5%

    minibatch_size = 32

    loss_fn = F.binary_cross_entropy

    history = []
    i_step = 0

    # pętla uczenia gradientowego
    for _ in range(n_steps):

        # TODO losowa paczka (mini-batch) danych o rozmiarze minibatch_size
        #  (użyj torch.randint do wylosowania indeksów przykładów
        idx = torch.randint(0, len(x), (minibatch_size, ))
        x_batch = x[idx]
        y_batch = y[idx]

        # TODO forward pass modelu + policzenie wartości funkcji kosztu (użyj loss_fn zdefiniowanego wyżej)
        y_pred = model.forward(x_batch)
        loss = loss_fn(y_pred, y_batch)

        # backward pass
        for p in model.parameters():
            p.grad = None  # gradienty są akumulowane, więc przed następnym krokiem trzeba je zresetować (specyfika pyTorch)
        loss.backward()  # autograd - policzenie gradientów metodą wstecznej propagacji - wypełnia pola .grad każdego
                         # z parametrów, który wpływa na wartość loss

        # TODO update params
        with torch.no_grad():
            for p in model.parameters():
                new_val = p - learning_rate * p.grad
                p.copy_(new_val)

        # Update lerning rate (Step-based)
        if i_step % step_size == 0 and i_step>0:
            learning_rate = step_learning_rate_schedule(
                initial_learning_rate,
                step_size,
                gamma,
                i_step)



        # track stats
        history.append(loss.log10().item())
        i_step += 1
        if i_step % 1000 == 0:
            print(f'Step={i_step}, lr={learning_rate:.6f}, loss={loss.item():.6f}')

    return model, history


def plot_history(history, smoothing_factor=0.99):
    plt.plot(history, c='b', alpha=0.5)
    if smoothing_factor is not None:
        smoothed_history = []; pv = history[0]
        for v in history:
            smoothed_history.append(smoothed := smoothing_factor*pv + (1.0-smoothing_factor)*v)
            pv = smoothed
        plt.plot(smoothed_history, c='b')
    plt.xlabel('Training step')
    plt.ylabel('Log loss')
    plt.show()


def classify_spirals(student_id, do_data_inspection=True, do_model_inpection=True, do_model_training=True,
                     load_trained_model=False):

    model_path = "my_model.pkl"

    # generacja danych
    n_examples = 1000
    gen = ClassificationTwoSpiralsData(degrees=900)
    x, y = gen.generate_data(n_examples=n_examples, noise=3.0, seed=student_id)

    # podział na zbiór uczący i testowy:
    n_train = int(0.8 * n_examples)
    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:], y[n_train:]

    if do_data_inspection:
        inspect_data(x_train, y_train, x_test=x_test, y_test=y_test)
        plot_data(x, y, x_test=x_test, y_test=y_test, plot_xy_range=[-1.2, 1.2])

    # konwersja danych do torch.Tensor
    x_train, y_train, x_test, y_test = [torch.from_numpy(x) for x in [x_train, y_train, x_test, y_test]]

    # utworzenie modelu
    model = TorchMultiLayerNetwork(
        n_in=2,
        n_hiddens=[16, 32, 32, 16],
        n_out=1)

    if load_trained_model:
        model.load_state_dict(torch.load(model_path))
        print(f"Wagi modelu załadowane z {model_path}")

    if do_model_inpection:
        print(model)
        print("Aktualne wagi:")
        for name, param in model.named_parameters():
            print(name, param.data)                 # początkowo wagi losowe
        print("Aktualne gradienty:")
        for name, param in model.named_parameters():
            print(name, param.requires_grad)        # czy dla tego parametru będzie potrzebne liczenie gradientu
            print(name, param.grad)                 # aktualna wartość gradientu

    print(f'Train accuracy: {evaluate_model(model, x_train, y_train) * 100:.02f}%')
    print(f'Test accuracy: {evaluate_model(model, x_test, y_test) * 100:.02f}%')

    if do_model_training:
        model, history = training(model,
                                  x_train,
                                  y_train)
        plot_history(history)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}.")

    # ewaluacja po treningu
    print(f'Train accuracy: {evaluate_model(model, x_train, y_train) * 100:.02f}%')
    print(f'Test accuracy: {evaluate_model(model, x_test, y_test) * 100:.02f}%')

    # test na całej przestrzeni wejść (z wizualizacją)
    x_grid = x_data_from_grid(min_xy=-1.2, max_xy=1.2, grid_size=1000)
    with torch.no_grad():
        y_pred_grid = model.forward(torch.from_numpy(x_grid))
    plot_data(x_train, y_train, x_test=x_test, y_test=y_test,
              plot_xy_range=[-1.2, 1.2], x_grid=x_grid, y_grid=y_pred_grid.numpy())


if __name__ == '__main__':

    student_id = 193511         # Twój numer indeksu, np. 102247
    torch.manual_seed(student_id)

    classify_spirals(student_id,
                     do_data_inspection=True,
                     do_model_inpection=True,
                     do_model_training=True,
                     load_trained_model=False)
