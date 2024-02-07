## How to Run

We provide the running scripts in `scripts`, which allow you to reproduce the results on our paper.

Change the path of `$DATA` and `$OUTPUT_DIR` in bash files (if not default), and run the following commands under the main directory.

### Base-to-New Generalization

All you need is `./scripts/Meta/b2n.sh` for training and testing.

`$DATASET` takes as input a dataset name, like `imagenet` or `caltech101`.

Below we provide examples of how to run MetaPrompt on base-to-new generalization.

#### **Train and Test on Base Classes** together with Test on New Classes:

- e.g. Caltech101: `bash ./scripts/Meta/b2n.sh caltech101`

### Domain Generalization
All you need is `./scripts/Meta/dg.sh` for training and testing of domain generalization.

`$DATASET` takes as input a dataset name, like `vlcs`.

`$SHOTS` takes as input the number of samples per class per domain, like 1 or 5.

Below we provide examples of how to run MetaPrompt on domain generalization.

#### **Training on ImageNet**:

`bash ./scripts/Meta/dg.sh office_home 5`
