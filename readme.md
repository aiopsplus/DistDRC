# Supplementary Material

This replication package contains supplementary material for the paper "DistDRC: Automated Annotation of Desirable Comments for Enhanced Code Review with Large Language Models". The package is organized as follows:

* `Data/`  The datasets used in this paper
  * `Data/CodeReviewerTrainDataset.xlsx` The original desiredness score results of the training dataset analysis for codereviewer.
  * `Data/CodeReviewerTestDataset.xlsx` The original desiredness score results of the testing dataset analysis for codereviewer.
  * `Data/OriginFinetuneDataset.jsonl` The training set for the automatic review comment generation task of LLaMA-Reviewer.
  * `Data/DesiredFinetuneDataset.jsonl` The training set for the automatic review comment generation task of Desiview4FT.
  * `Data/AlignmentDataset.jsonl` The training set for the automatic review comment generation task of Desiview4FA.
  * `Data/TestDataset.jsonl`  The testing set for the automatic review comment generation task.
* `Scorer/` Source code for scoring desiredness code fixes.
* `Finetune/` The fine-tuning code for the review comment generation task.
* `Alignment/` The alignment code for the review comment generation task.
