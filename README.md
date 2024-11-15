# Context-aware Negative Sampling

---

* raw dataset directory: `./raw/`
    * put [CARCA/Data](https://github.com/ahmedrashed-ml/CARCA) as `./raw/CARCA/`
    * `./raw/ml20m`
* data directory: `./data/`

For preprocessing, anaconda environment with `requirements.txt` installed is recommended.

```bash
python preprocess.py prepare --dname game
python preprocess.py prepare --dname beauty
python preprocess.py prepare --dname fashion
python preprocess.py prepare --dname men
python preprocess.py prepare --dname ml1m
python preprocess.py prepare --dname ml20m
python preprocess.py prepare --dname dressipi
python preprocess.py prepare --dname steam
python preprocess.py prepare --dname retailrocket
python preprocess.py prepare --dname diginetica

python preprocess.py count_stats
```

For training and evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python entry.py ml1m
```

Easy tensorboard:

```bash
./scripts/tboard.sh
```
