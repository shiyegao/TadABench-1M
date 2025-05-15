from typing import Tuple
from torch.utils.data import Dataset
from datasets import load_dataset


CODON2AA = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

AA2CODON = {}
for k, v in CODON2AA.items():
    if v not in AA2CODON:
        AA2CODON[v] = []
    AA2CODON[v].append(k)


def DNA2AA(DNA: str):
    assert len(DNA) % 3 == 0, f"Invalid DNA length: {len(DNA)}"
    return "".join(CODON2AA[DNA[i : i + 3]] for i in range(0, len(DNA), 3))


def DNA2RNA(DNA: str):
    return DNA.replace("T", "U")


def modality_map(seq_type: str, seq: str):
    if seq_type == "AA":
        seq = DNA2AA(seq)
    if seq_type == "RNA":
        seq = DNA2RNA(seq)
    return seq


class RegressionDataset(Dataset):
    def __init__(self, cfg, split: str = ""):
        self.seq_type = cfg.seq_type
        self.split = split
        self.dataset_name = cfg.huggingface_dataset
        self.length = cfg.length
        self.return_seq = getattr(cfg, "return_seq", False)
        self.normalize_label = getattr(cfg, "normalize_label", False)

        self.load_data()

    def load_data(self):
        data = load_dataset(
            self.dataset_name, split=f"all.{self.seq_type}.{self.split}"
        )
        self.data = data["Sequence"]
        self.labels = data["Value"]

        if self.normalize_label:
            self.max_label = max(self.labels)
            self.min_label = min(self.labels)
            self.labels = [
                (label - self.min_label) / (self.max_label - self.min_label)
                for label in self.labels
            ]
            print(
                f"Normalizing labels from [{self.min_label}, {self.max_label}] to [{min(self.labels)}, {max(self.labels)}]"
            )

        self.num_samples = len(self.data)
        self.seqs = set(self.data)
        print(f"Loaded {len(self.data)} sequences ({len(self.seqs)} unique)")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Tuple[Tuple[str], Tuple[str]]:
        data, label = self.data[idx], self.labels[idx]

        if self.return_seq:
            return (data, self.data[idx]), label
        else:
            return data, label
