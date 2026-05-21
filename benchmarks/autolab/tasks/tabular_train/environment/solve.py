import numpy as np
import torch
import torch.nn as nn

SEED    = 42
EPOCHS  = 30
LR      = 0.01
HIDDEN  = 32

# Architecture dimensions are stored in the model dict so predict() can
# reconstruct the network without hard-coding them here.


def _build_net(n_in, hidden, n_out):
    return nn.Sequential(
        nn.Linear(n_in, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_out),
    )


def train(X_train, y_train):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    n_in  = X_train.shape[1]
    n_out = int(y_train.max()) + 1

    X = torch.from_numpy(X_train.astype(np.float32))
    y = torch.from_numpy(y_train.astype(np.int64))

    net = _build_net(n_in, HIDDEN, n_out)
    opt = torch.optim.SGD(net.parameters(), lr=LR)

    for _ in range(EPOCHS):
        logits = net(X)
        loss   = nn.functional.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    state = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
    state["_n_in"]    = np.array([n_in],    dtype=np.int64)
    state["_hidden"]  = np.array([HIDDEN],  dtype=np.int64)
    state["_n_out"]   = np.array([n_out],   dtype=np.int64)
    return state


def predict(model, X):
    n_in   = int(model["_n_in"][0])
    hidden = int(model["_hidden"][0])
    n_out  = int(model["_n_out"][0])

    net = _build_net(n_in, hidden, n_out)
    sd  = {k: torch.from_numpy(v) for k, v in model.items() if not k.startswith("_")}
    net.load_state_dict(sd)
    net.eval()

    with torch.no_grad():
        logits = net(torch.from_numpy(X.astype(np.float32)))
    return logits.argmax(dim=1).numpy().astype(np.int64)
