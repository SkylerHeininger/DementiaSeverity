import torch
import torch.nn as nn

MIN_SEGMENTS = 10
loaded = {}

class SessionCNN(nn.Module):
    # Same layers and attribute names as in the training script, so the saved weights load into it
    def __init__(self, n_roles, n_groups, speaker_dim, group_dim, n_filters, kernel, layers, dropout):
        super().__init__()
        self.speaker = nn.Embedding(n_roles, speaker_dim)
        self.group = nn.Embedding(n_groups + 1, group_dim)
        convs, dim = [], speaker_dim + group_dim

        for _ in range(layers):
            convs.append(nn.Conv1d(dim, n_filters, kernel, padding=kernel // 2))
            dim = n_filters

        self.convs = nn.ModuleList(convs)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * n_filters, 1)

    def forward(self, speaker, group, mask):
        x = torch.cat([self.speaker(speaker), self.group(group)], dim=2)
        maskf = mask.float()
        h = x.transpose(1, 2)

        for conv in self.convs:
            h = torch.relu(conv(h)) * maskf[:, None, :]

        top = h.masked_fill(~mask[:, None, :], float("-inf")).amax(dim=2)
        mean = h.sum(dim=2) / maskf.sum(dim=1, keepdim=True)

        return self.head(self.drop(torch.cat([top, mean], dim=1))).squeeze(1)


def load(path):
    stamp = path.stat().st_mtime
    if loaded.get("path") != path or loaded.get("stamp") != stamp:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        net = SessionCNN(len(saved["roles"]), len(saved["da_groups"]), **saved["config"])
        net.load_state_dict(saved["state_dict"])
        net.eval()
        loaded.update(path=path, stamp=stamp, net=net, saved=saved)

    return loaded["net"], loaded["saved"]


def sequence(saved, da):
    roles = {role: i for i, role in enumerate(saved["roles"])}
    groups = {group: i for i, group in enumerate(saved["da_groups"])}
    segments = [s for s in da["segments"] if s["role"] in roles]
    if len(segments) < MIN_SEGMENTS:
        raise ValueError(f"needs at least {MIN_SEGMENTS} dialogue act segments, this visit has {len(segments)}")

    speaker = [roles[s["role"]] for s in segments]
    group = [groups[saved["da_to_group"].get(str(s["label"]).strip(), "other")] for s in segments]
    return speaker, group


def predict(path, da):
    net, saved = load(path)
    speaker, group = sequence(saved, da)
    batch = (torch.tensor([speaker]), torch.tensor([group]), torch.ones(1, len(speaker), dtype=torch.bool))

    with torch.no_grad():
        standardized = float(net(*batch)[0])

    return standardized * saved["std"] + saved["mean"]
