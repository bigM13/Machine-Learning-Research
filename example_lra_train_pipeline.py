import numpy as np
import torch
import torch.nn.functional as F
from my_models import (ManualRNNClassifier, ChaosUnitClassifier, CustomClassifier, ManualRNNReluClassifier, LinearRNNClassifier, LinearResidualRNNClassifier, StackedManualRNNClassifier, RNNWithSigmoid, RNNWord2Vec, StackedRNNWord2Vec)
from newer_custom_models import (RNNMockingTransformer, MyMambaModel, ProjToMultiLayerRNN, TransformerClassifier, RNNMockingMurmurHash3, SimpleTransformerBaseline)
from tqdm import tqdm
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from ml_collections import ConfigDict
from lra_config import (get_listops_config, get_cifar10_config, get_text_classification_config, get_pathfinder32_config, get_pathfinder64_config, get_pathfinder128_config, get_pathfinder256_config, get_document_retrieval_config)
from lra_datasets import (ListOpsDataset, Cifar10Dataset, ImdbDataset, Pathfinder32Dataset, Pathfinder64Dataset, Pathfinder128Dataset, Pathfinder256Dataset, DocumentRetrievalDataset)
from argparse import ArgumentParser

def dict_to_device(inputs, device):
    return {key: inputs[key].to(device) for key in inputs}

#def transformers_collator(batch):
#    input_ids = torch.stack([item[0]["input_ids"].squeeze(0) for item in batch])  # [B, 1024]
#    attn_mask = torch.stack([item[0]["attention_mask"].squeeze(0) for item in batch])  # [B, 1024]
#    labels = torch.stack([item[1] for item in batch])

    # normalize to 0–1
#    input_ids = input_ids.float() / 255.0
#    input_ids -= 0.5

#    return {"input_ids": input_ids, "attention_mask": attn_mask}, labels

def transformers_collator(samples):
    imgs, labels = zip(*samples)

    # If tokenizer returns dicts (with input_ids and possibly attention_mask)
    if isinstance(imgs[0], dict):
        batch_imgs = torch.stack([img["input_ids"].float() / 255.0 for img in imgs])
        batch_imgs -= 0.5  # shift roughly to [-0.5, 0.5]

        attention_mask = torch.stack([img["attention_mask"].float() for img in imgs])
    else:
        # If tokenizer returns plain tensors
        batch_imgs = torch.stack([img["input_ids"].float() / 255.0 for img in imgs])
        batch_imgs -= 0.5  # shift roughly to [-0.5, 0.5]

        attention_mask = torch.ones(batch_imgs.shape[:2], dtype=torch.float32)  # default mask

    labels = torch.stack(labels).long()

    return {"input_ids": batch_imgs, "attention_mask": attention_mask}, labels



def accuracy_score(outp, target):
    assert outp.dim() == 2, f"accuracy expects 2D outputs, got {outp.shape}"
    assert target.dim() == 1, f"accuracy expects 1D target, got {target.shape}"
    return (torch.argmax(outp, dim=-1) == target).sum().item() / len(target)

# consts
OUTPUT_DIR = "output_dir/"
deepspeed_json = "ds_config.json"

TASKS = {
    'listops': ConfigDict(dict(dataset_fn=ListOpsDataset, config_getter=get_listops_config)),
    'cifar10': ConfigDict(dict(dataset_fn=Cifar10Dataset, config_getter=get_cifar10_config)),
    'imdb': ConfigDict(dict(dataset_fn=ImdbDataset, config_getter=get_text_classification_config)),
    'pathfinder32': ConfigDict(dict(dataset_fn=Pathfinder32Dataset,config_getter=get_pathfinder32_config)),
    'pathfinder64': ConfigDict(dict(dataset_fn=Pathfinder64Dataset,config_getter=get_pathfinder64_config)),
    'pathfinder128': ConfigDict(dict(dataset_fn=Pathfinder128Dataset,config_getter=get_pathfinder128_config)),
    'pathfinder256': ConfigDict(dict(dataset_fn=Pathfinder256Dataset,config_getter=get_pathfinder256_config)),
    'documentretrieval': ConfigDict(dict(dataset_fn=DocumentRetrievalDataset,config_getter=get_document_retrieval_config)),
}


# model loading
def get_model(config, model_config, pretrained_weights=None):
    model = SimpleTransformerBaseline(
        #embedding=pretrained_weights,
        vocab_size=model_config.vocab_size,
        #hidden_size=model_config.hidden_size,
        #num_classes=model_config.num_labels,
        #max_length=1,
        #embed_dim=model_config.embed_dim,
        #num_layers=2,
        #num_heads=4,
        #hidden_multiplier=model_config.hidden_multiplier if 'hidden_multiplier' in model_config else 1
        input_dim=1024,
        #proj_dim=1792,
        #dim_feedforward=128,
        hidden_dim=256, # 2048 super sucessful
        #model_dim=256,
        #mamba_state_dim=128
    )
    return model

def train(model, dataloader, optimizer, scheduler, device, grad_accum_steps=1):
    """Run one epoch of training."""
    model.train()
    running_loss, running_acc, steps = 0.0, 0.0, 0
    
    pbar = tqdm(dataloader, desc="Train", leave=False)
    for i, batch in enumerate(pbar):
        x, target = batch
        x = dict_to_device(x, device)
        target = target.to(device)
        #inputs = dict_to_device(inputs, device)
        #target = target.to(device)
        
        outputs = model(**x)
        #if return_hidden:
        #    outputs, hidden = model(x, return_hidden=True)
        #else:
        #    outputs = model(x, return_hidden=False)

        loss = F.cross_entropy(outputs, target.long()) / grad_accum_steps
        loss.backward()

        # DEBUG: check gradient flow
        #if i == 0:  # only print once to avoid spam
        #    for name, p in model.named_parameters():
        #        if p.grad is None:
        #            print(f"{name} has no grad (in TRAIN)")
        #        else:
        #            print(f"{name} grad mean: {p.grad.abs().mean().item():.6f}")


        if (i + 1) % grad_accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        #with torch.no_grad():
        #    acc = accuracy_score(outputs, target)
        preds = outputs.argmax(dim=1)  # get predicted class index (0 or 1)
        acc = (preds == target).float().mean().item()


        running_loss += loss.item() * grad_accum_steps
        running_acc += acc
        steps += 1
        pbar.set_postfix(loss=running_loss / steps, acc=running_acc / steps)

    return running_loss / steps, running_acc / steps


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model on validation/test data."""
    model.eval()
    running_loss, running_acc, steps = 0.0, 0.0, 0

    pbar = tqdm(dataloader, desc="Eval", leave=False)
    for batch in pbar:
        x, target = batch
        x = dict_to_device(x, device)
        target = target.to(device)
        #inputs = dict_to_device(inputs, device)
        #target = target.to(device)

        outputs = model(**x)
        loss = F.cross_entropy(outputs, target.long())
        preds = outputs.argmax(dim=1)  # get predicted class index (0 or 1)
        acc = (preds == target).float().mean().item()
        #acc = accuracy_score(outputs, target)
        running_loss += loss.item()
        running_acc += acc
        steps += 1
        pbar.set_postfix(loss=running_loss / steps, acc=running_acc / steps)

    return running_loss / steps, running_acc / steps

from train_utils import create_learning_rate_scheduler

def main():
    parser = ArgumentParser()
    parser.add_argument("--task", default="pathfinder32", choices=TASKS.keys(), help="choose an LRA dataset")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    task = TASKS[args.task]
    config, model_config = task.config_getter()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # dataset + loaders
    train_set = task.dataset_fn(config, split="train")
    eval_set = task.dataset_fn(config, split="eval")
    test_set = task.dataset_fn(config, split="test")

    #inspect_raw_dataset(eval_set)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, collate_fn=transformers_collator, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=config.batch_size, collate_fn=transformers_collator)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, collate_fn=transformers_collator)

    model = get_model(config, model_config)
    model.to(device)

    # optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
    scheduler = config.lr_scheduler(optimizer)

    #####
    ####
    # === DIAGNOSTIC CHECK ===
    print("\n[Dataset sanity check]")
    batch = next(iter(train_loader))
    x, y = batch
    print("x keys:", x.keys())
    print("input_ids mean/std:", x["input_ids"].mean().item(), x["input_ids"].std().item())
    print("attention_mask unique:", x["attention_mask"].unique())
    print("labels unique:", torch.unique(y, return_counts=True))




    ####
    #####

    # train loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, scheduler, device, grad_accum_steps=config.get("gradient_accumulation_steps", 1))
        eval_loss, eval_acc = evaluate(model, eval_loader, device)

        print(f"[Epoch {epoch}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f}")

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"TEST DATA RESULTS: "
              f"eval_loss={test_loss:.4f} eval_acc={test_acc:.4f}")

    # save model if you want
    torch.save(model.state_dict(), f"{args.task}_model.pth")
    

if __name__ == "__main__":
    main()