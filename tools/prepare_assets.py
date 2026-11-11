#!/usr/bin/env python3
from pathlib import Path
import argparse,json,hashlib
from huggingface_hub import hf_hub_download, HfApi
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--xcomet-revision');a=p.parse_args()
repo='Unbabel/XCOMET-XL'
revision=a.xcomet_revision or HfApi().model_info(repo).sha
# Authentication uses the caller's existing HF_TOKEN. Accept upstream terms in your account first.
path=Path(hf_hub_download(repo,'checkpoints/model.ckpt',revision=revision,local_dir=root/'models/xcomet',cache_dir=root/'cache/downloads'))
h=hashlib.sha256()
with path.open('rb') as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
(root/'provenance/XCOMET_DOWNLOAD.json').write_text(json.dumps(dict(repo=repo,revision=revision,file=str(path.relative_to(root)),sha256=h.hexdigest(),original_paper_weight_hash_available=False),indent=2))
print('Downloaded XCOMET; original paper checkpoint checksum was unavailable, so equality is not asserted.')
