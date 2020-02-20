# CLEF-HIPE-2020-scorer

CLEF-HIPE-2020-scorer is a python module for evaluating Named Entity Recognition and Classification (NER) models and Named Entity Linking (NEL) as defined in the [CLEF-HIPE-2020 Shared Task](https://impresso.github.io/CLEF-HIPE-2020//). The format of the data is similar to [CoNLL-U](https://universaldependencies.org/format.html), yet a token may have more than one named entity annotations. Different annotations are recorded in different columns, which are evaluated separately (for more detail on system input format, refer to the [HIPE participation guidelines](https://zenodo.org/record/3604238)). 

The NERC evaluation goes beyond a token-based schema and considers entities as the unit of reference. An entity is defined as being of a particular type with a token onset as well as an offset. Following this definition, various NERC system outcomes may exist regarding correct entity types and boundaries when comparing the system response to the gold standard. The task of named entity evaluation is described in detail in the [original blog post](http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/) by [David Batista](https://github.com/davidsbatista). The present scorer extends the code in the [original repository](https://github.com/davidsbatista/NER-Evaluation), which accompanied the blog post.

### Metrics

#### NERC

NERC is evaluated in terms of **macro and micro Precision, Recall, F1-measure**. Two evaluation scenarios are considered: **strict** (exact boundary matching) and **relaxed** (fuzzy boundary matching). 

Each column is evaluated independently, according to the following metrics:

- Micro average P, R, F1 at entity level (not at token level), i.e. consideration of all true positives, false positives, true negatives and false negative over all documents. 
  - strict (exact boundary matching) and fuzzy (at least 1 token overlap).
  - separately per type and cumulative for all types.
  
- Document-level macro average P, R, F1 at entity level (not on token level). i.e. average of separate micro evaluation on each individual document.
  - strict and fuzzy
  - separately per type and cumulative for all types
  
Our definition of macro differs from the usual one, and macro measures are computed as aggregates on document-level instead of entity-type level. Specifically, macro measures average the corresponding micro scores across all the documents, accounting for (historical) variance in document length and not for class imbalances.


Note that in the strict scenario, predicting wrong boundaries leads to severe punishment of one false negative (entity present in the gold standard but not predicted by the system) and one false positive (predicted entity by the system but not present in the gold standard). Although this may be severe, we keep this metric in line with [CoNLL](https://www.clips.uantwerpen.be/conll2000/chunking/output.html) and refer to the fuzzy scenario if the boundaries of an entity are considered as less important.

#### EL

The evaluation for NEL works similarly as for NERC. The link of an entity is interpreted as a label. As there is no IOB-tagging, a consecutive row of identical links is considered as a single entity. In terms of boundaries, NEL is only evaluated according to the fuzzy scenario. Thus, to get counted as correct, the system response needs only one overlapping link label with the gold standard. 

The Slot Error Rate (SER) is dropped for the shared task evaluation.


## Scorer
To evaluate the predictions of your system on the dev set, you can run the following command:

```python clef_evaluation.py --ref DEVSET.tsv --pred PREDICTIONS.tsv --task TASK```

There are three different evaluation modes available: `nerc_coarse`, `nerc_fine`, `nel`. Depending on the task, the script performs the evaluation for the corresponding columns and evaluation scenarios automatically. 

The script expects both files (system responses and gold standard) to have a similar structure (columns) as well as similar content (same tokens in the same order). In cases of mismatches, the evaluation fails and outputs information about the issue. Specifically, the current version of the script requires an identical segmentation of the predictions based on `# document_id` and `# segment_iiif_link`. Other comment lines starting with a `#` may be omitted.

Unless you define otherwise with the `--skip_check` argument, the evaluation script enforces the following convention on the file name (see [HIPE Shared Task Participation Guidelines](https://zenodo.org/record/3604238), p. 13): `TEAMNAME_TASKBUNDLEID_LANG_RUNNUMBER.tsv`

### Output

The evaluation script outputs two files in the same directory where the prediction file is located: `results_TASK.tsv` and `results_TASK_all.json`

The condensed tsv-report (`results_TASK.tsv`) contains all the measures that are relevant for the official shared task evaluation. The report has the following structure:

| System                               | Evaluation               | Label | P    | R    | F1   | F1_std | P_std | R_std | TP   | FP   | FN   |
| ------------------------------------ | ------------------------ | ----- | ---- | ---- | ---- | ------ | ----- | ----- | ---- | ---- | ---- |
| TEAMNAME_TASKBUNDLEID_LANG_RUNNUMBER | NE-FINE-COMP-micro-fuzzy | ALL   |      |      |      |        |       |       |      |      |      |

Cells may be empty in case they are not defined or provide only redundant information. The column `Evaluation` refers to the evaluated column and defines the measures P, R, F1, etc. It has the following structure: `COL_NAME-{micro/macro_doc}-{fuzzy-strict}`. This schema makes it easy to filter for a particular metric with `grep`.

The detailed json-report (`results_TASK_all.json`) contains all the measures that are available (see below).

Moreover, the figures are computed in aggregate (Label = ALL) and per entity-type.

### Available figures and metrics
For any of the evaluation schemes, the evaluation provides the following figures:

- `correct`
- `incorrect`
- `partial`
- `missed`
- `spurious`
- `possible` (=number of annotations in the gold standard)
- `actual` (=number of annotations predicted by the system)
- `TP`
- `FP`
- `FN`
- `P_micro`
- `R_micro`
- `F1_micro`
- `P_macro_doc`
- `R_macro_doc`
- `F1_macro_doc`
- `P_macro_doc_std`
- `R_macro_doc_std`
- `F1_macro_doc_std`
- `P_macro`
- `R_macro`
- `F1_macro`
- `F1_macro (recomputed from P & R)`



## Baseline
We report baseline scores for Named Entity Recognition and Classification (NERC) using a trained basic CRF model. The model uses surface information only and dismisses the segmentation structure as it treats any particular document as a single, long sentence. The model is trained on the official training data and evaluated on the development set.

To train and evaluate a baseline model for the *NERC Coarse-grained* task, you can run the following command:

```python baseline.py --train TRAIN.tsv --dev DEV.tsv  --pred OUTPUT.tsv --cols NE-COARSE-LIT,NE-COARSE-METO --eval nerc_coarse```

`--cols` defines the columns for which baselines are trained.

`--eval` defines the task on which the baseline is evaluated.


The script outputs three files: a file with the predictions of the baseline system, a condensed report and a detailed report (see evaluation).

Please note that there is no baseline for named entity linking (NEL).

## Acknowledgments
Our evaluation module is based on David Batista's [NER-Evaluation](https://github.com/davidsbatista/NER-Evaluation).

