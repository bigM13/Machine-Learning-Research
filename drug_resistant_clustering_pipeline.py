"""
Exploratory clustering of WGS reads to identify resistance-associated patterns.

- Input: FASTA files (one per isolate)
- Method: k-mer encoding + autoencoder + clustering
- Output: cluster summaries and resistance motif signals

This pipeline is hypothesis-generating, not confirmatory.
"""

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from collections import defaultdict, Counter
from sklearn.cluster import MiniBatchKMeans

############################################
# CONFIGURATION
############################################

ISOLATE_FASTAS = {
    "isolate_main": "data/dataset.fasta",
    "background_1": "data/K22_sequence.fasta",
    "background_2": "data/K31_sequence.fasta",
}

K = 5
LATENT_DIM = 32
BATCH_SIZE = 512
EPOCHS = 5
N_CLUSTERS = 6
MAX_READS_PER_ISOLATE = 200_000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

############################################
# k-mer encoding
############################################

BASES = ["A", "C", "G", "T"]

def generate_kmer_index(k):
    kmers = ["".join(p) for p in __product(BASES, k)]
    return {kmer: i for i, kmer in enumerate(kmers)}

def __product(chars, k):
    if k == 1:
        return chars
    return [c + p for c in chars for p in __product(chars, k - 1)]

KMER_INDEX = generate_kmer_index(K)
INPUT_DIM = len(KMER_INDEX)

def encode_kmers(seq):
    vec = np.zeros(INPUT_DIM, dtype=np.float32)
    seq = seq.upper()
    for i in range(len(seq) - K + 1):
        kmer = seq[i:i+K]
        if "N" in kmer:
            continue
        idx = KMER_INDEX.get(kmer)
        if idx is not None:
            vec[idx] += 1
    if vec.sum() > 0:
        vec /= vec.sum()
    return vec

def gc_content(seq):
    seq = seq.upper()
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)

############################################
# FASTA streaming dataset
############################################

class FastaDataset(IterableDataset):
    def __init__(self, fasta_path, isolate_id, max_reads):
        self.fasta_path = fasta_path
        self.isolate_id = isolate_id
        self.max_reads = max_reads

    def __iter__(self):
        with open(self.fasta_path) as f:
            seq = ""
            count = 0
            for line in f:
                if line.startswith(">"):
                    if seq:
                        yield {
                            "kmer": encode_kmers(seq),
                            "gc": gc_content(seq),
                            "isolate": self.isolate_id
                        }
                        count += 1
                        if count >= self.max_reads:
                            return
                    seq = ""
                else:
                    seq += line.strip()
            if seq and count < self.max_reads:
                yield {
                    "kmer": encode_kmers(seq),
                    "gc": gc_content(seq),
                    "isolate": self.isolate_id
                }

############################################
# Autoencoder model
############################################

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

############################################
# LOAD DATA
############################################

print("Loading data...")

all_records = []
for isolate, fasta in ISOLATE_FASTAS.items():
    ds = FastaDataset(fasta, isolate, MAX_READS_PER_ISOLATE)
    for record in ds:
        all_records.append(record)

kmer_matrix = np.vstack([r["kmer"] for r in all_records])
gc_values = np.array([r["gc"] for r in all_records])
isolate_labels = [r["isolate"] for r in all_records]

############################################
# TRAIN AUTOENCODER
############################################

print("Training autoencoder...")

model = AutoEncoder(INPUT_DIM, LATENT_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

dataset_tensor = torch.tensor(kmer_matrix, dtype=torch.float32)
loader = DataLoader(dataset_tensor, batch_size=BATCH_SIZE, shuffle=True)

model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for batch in loader:
        batch = batch.to(DEVICE)
        recon, _ = model(batch)
        loss = loss_fn(recon, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {epoch_loss:.4f}")

############################################
# EMBEDDING + CLUSTERING
############################################

print("Embedding and clustering...")

model.eval()
with torch.no_grad():
    embeddings = model.encoder(dataset_tensor.to(DEVICE)).cpu().numpy()

kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42)
cluster_ids = kmeans.fit_predict(embeddings)

############################################
# CLUSTER SUMMARIES
############################################

cluster_summary = defaultdict(lambda: {
    "count": 0,
    "gc": [],
    "isolates": Counter(),
    "kmer_vectors": []
})

for cid, iso, gc, vec in zip(cluster_ids, isolate_labels, gc_values, kmer_matrix):
    cs = cluster_summary[cid]
    cs["count"] += 1
    cs["gc"].append(gc)
    cs["isolates"][iso] += 1
    cs["kmer_vectors"].append(vec)

############################################
# RESISTANCE MOTIF HEURISTICS
############################################

KNOWN_RESISTANCE_MOTIFS = {
    "blaKPC": ["TGGCG", "CGTGG"],
    "blaNDM": ["GGGCG", "GATCG"],
    "blaOXA": ["ACGAA", "TCGAC"],
    "blaVIM": ["GCGCG", "CGCGG"]
}

print("\n=== CLUSTER REPORT ===\n")

global_kmer_mean = np.mean(kmer_matrix, axis=0)

for cid, info in cluster_summary.items():
    mean_gc = np.mean(info["gc"])
    mean_kmer = np.mean(info["kmer_vectors"], axis=0)
    enriched = mean_kmer - global_kmer_mean
    top_kmers = np.argsort(enriched)[-25:]

    hits = set()
    for gene, motifs in KNOWN_RESISTANCE_MOTIFS.items():
        for m in motifs:
            idx = KMER_INDEX.get(m)
            if idx is not None and idx in top_kmers:
                hits.add(gene)

    print(f"Cluster {cid}")
    print(f"  Reads: {info['count']}")
    print(f"  Mean GC: {mean_gc:.3f}")
    print(f"  Isolate distribution: {dict(info['isolates'])}")
    print(f"  Resistance signals: {hits if hits else 'None detected'}")
    print("")

print("Analysis complete.")