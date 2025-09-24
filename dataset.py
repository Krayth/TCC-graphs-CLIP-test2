"""1. `scale` modificado para 0.8 e `resize` para 224, para adicionar contexto"""
import random

# Gambiarra para importar as funções de dataset feitas anteriormente

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.v2 as transf

import json
import random
from pathlib import Path

import torch
import torchvision.transforms.v2 as transf
from PIL import Image
from torch.utils.data import Dataset


class Subset(Dataset):
    """Subset de outro dataset usando índices escolhidos."""

    def __init__(self, ds, indices, transform=None):
        self.ds = ds
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        img, text = self.ds[self.indices[idx]]
        if self.transform is not None:
            img = self.transform(img)
        return img, text

    def __len__(self):
        return len(self.indices)


class GraphDataset(Dataset):
    """Dataset com imagens de grafos e suas listas de arestas."""

    def __init__(self, root, json_file, transforms=None):
        """
        Args:
            root: pasta com imagens
            json_file: caminho do json com as listas de arestas
            transforms: transformações para aplicar na imagem
        """
        self.root = Path(root)
        self.transforms = transforms

        with open(json_file, "r") as f:
            self.graph_data = json.load(f)

        self.ids = list(self.graph_data.keys())

    def __getitem__(self, idx, apply_transform=True):
        graph_id = self.ids[idx]
        edges = self.graph_data[graph_id]

        img_path = self.root / f"{graph_id}.png"
        image = Image.open(img_path).convert("RGB")

        # transforma lista de arestas em string
        text_edges = " ".join([f"{u}-{v}" for u, v in edges])

        if self.transforms and apply_transform:
            image = self.transforms(image)

        return image, text_edges

    def __len__(self):
        return len(self.ids)


class TransformsTrain:
    """Transformações de treinamento"""

    def __init__(self, resize_size=224):
        transforms = transf.Compose([
            transf.PILToTensor(),
            transf.RandomResizedCrop(size=(resize_size, resize_size),
                                     scale=(0.5, 1.), ratio=(0.7, 1.3), antialias=True),
            transf.RandomHorizontalFlip(),
            transf.ToDtype(torch.float32),
            transf.Normalize(mean=(122.7, 114.6, 100.9),
                             std=(59.2, 58.4, 59.0))
        ])
        self.transforms = transforms

    def __call__(self, img):
        return self.transforms(img)


class TransformsEval:
    """Transformações de validação"""

    def __init__(self):
        transforms = transf.Compose([
            transf.PILToTensor(),
            transf.Resize(size=256, antialias=True),
            transf.CenterCrop(size=224),
            transf.ToDtype(torch.float32),
            transf.Normalize(mean=(122.7, 114.6, 100.9),
                             std=(59.2, 58.4, 59.0))
        ])
        self.transforms = transforms

    def __call__(self, img):
        return self.transforms(img)


def unormalize(img):
    """Reverte normalização para visualização."""
    img = img.permute(1, 2, 0)
    mean = torch.tensor([122.7, 114.6, 100.9])
    std = torch.tensor([59.2, 58.4, 59.0])
    img = img * std + mean
    img = img.to(torch.uint8)
    return img


def get_dataset(image_root, json_path, split=0.2, resize_size=224):
    """Cria splits de treino/validação para grafos.

    Args:
        image_root: diretório das imagens
        json_path: caminho do json de arestas
        split: fração de validação
        resize_size: tamanho de resize/crop
    """
    ds = GraphDataset(image_root, json_path)
    n = len(ds)
    n_valid = int(n * split)

    indices = list(range(n))
    random.seed(42)
    #random.shuffle(indices)

    ds_train = Subset(ds, indices[n_valid:], TransformsTrain(resize_size))
    ds_valid = Subset(ds, indices[:n_valid], TransformsEval())

    return ds_train, ds_valid


def wrap_text(text):
    """Função para quebrar o texto em linhas. Usada apenas para visualização
    dos dados.
    """
    text_split = text.split()
    for idx in range(len(text_split)):
        if (idx+1)%4==0:
            text_split[idx] += "\n"
        else:
            text_split[idx] += " "
    wrapped_text = "".join(text_split)

    return wrapped_text

def show_items(ds):

    inds = torch.randint(0, len(ds), size=(12,))
    items = [ds[idx] for idx in inds]

    fig, axs = plt.subplots(2, 6, figsize=(12,5))
    axs = axs.reshape(-1)
    for idx in range(12):
        image, caption = items[idx]
        caption = wrap_text(caption)
        axs[idx].imshow(image.permute(1, 2, 0)/255.)
        axs[idx].set_title(caption, loc="center", wrap=True)
    fig.tight_layout()

def collate_fn(batch):
    """Concatena imagens, mas não os textos"""
    images, texts = list(zip(*batch))
    batched_imgs = torch.stack(images, 0)

    return batched_imgs, texts