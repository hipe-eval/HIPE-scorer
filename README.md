# CLEF-HIPE-2020-scorer

CLEF-HIPE-2020-scorer is a python module for evaluating Named Entity Recognition and Classification (NER) models and Named Entity Linking as defined in the [CLEF-HIPE-2020 Shared Task](https://impresso.github.io/CLEF-HIPE-2020//). The format of the data is similar to [CoNLL-U](https://universaldependencies.org/format.html), yet a token may have more than one named entity annotations. Different annotations are recorded in different columns, which are evaluated separately. 

The NERC evaluation goes beyond a token-based schema and considers entities as the unit of reference. An entity is defined as being of a particular type with a token onset as well as an offset. Following this definition, NERC may have a multitude of outcomes regarding correct entity types and boundaries when comparing the system response to the gold standard. The task of NERC is described in detail in the [original blog post](http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/) by [David Batista](https://github.com/davidsbatista) and extends the code in the [original repository](https://github.com/davidsbatista/NER-Evaluation), which accompanied the blog post.

NERC will be  evaluated in terms of macro and micro Precision, Recall, F1-measure. Two evaluation scenarios will be considered: strict (exact boundary matching) and relaxed (fuzzy boundary matching). 

Other than the regular definition, macro measures are not computed to balance the distribution of classes. Macro measures average the corresponding micro scores across all the documents, accounting for (historical) variance in document length.

In the strict scenario, predicting wrong boundaries leads to severe punishment of one false negative (entity found in the gold standard not predicted by the system) and false positive (predicted entity not found in gold standard). Although this may be unfortunate, we keep this metric in line with [CoNLL](https://www.clips.uantwerpen.be/conll2000/chunking/output.html) and refer to the fuzzy scenario if the boundaries of an entity are considered as less important.

The evaluation for NEL works similarly as for NERC. The link of an entity is interpreted as a label. As there is no IOB-tagging, a consecutive row of identical links is considered as a single entity. NEL is only evaluated according to the fuzzy scenario. Thus, to get counted as correct,  the system response needs only one overlapping link label with the gold standard. 

The Slot Error Rate (SER) is dropped for the shared task evaluation.


## Usage
To evaluate the predictions of your system, you can run the following command:

```python clef_evaluation.py --ref DEVSET.tsv --pred PREDICTIONS.tsv --task TASK```

There are three different evaluation modes available: `nerc_coarse`, `nerc_fine`, `nel`. Depending on the task, the script performs the evaluation for the corresponding columns and evaluation scenarios automatically. 

The script expects both files to have a similar structure. In cases of mismatches, the evaluation fails and outputs information about the issue. Specifically, the current version of the script requires an identical segmentation of the predictions based on `# document_id` and `# segment_iiif_link`. Other comment lines starting with a `#` may be omitted.

Unless you define otherwise with the `--skip_check` argument, the evaluation script enforces the following convention on the file name: `TEAMNAME_TASKBUNDLEID_LANG_RUNNUMBER.tsv`

### Output

The evaluation script outputs two files in the same directory where the prediction file is located: `results_TASK.tsv`, `results_TASK_all.json`

The condensed tsv-report contains all the measures that are relevant for the official shared task evaluation. The report has the following structure:

| System                               | Evaluation               | Label | P    | R    | F1   | F1_std | P_std | R_std | TP   | FP   | FN   |
| ------------------------------------ | ------------------------ | ----- | ---- | ---- | ---- | ------ | ----- | ----- | ---- | ---- | ---- |
| TEAMNAME_TASKBUNDLEID_LANG_RUNNUMBER | NE-FINE-COMP-micro-fuzzy | ALL   |      |      |      |        |       |       |      |      |      |

Cells may be empty in cases when they are not defined or provide only redundant information. The column `Evaluation` defines refers to the evaluated column and specifies the measure P, R, F1, etc. It has the following structure: `COL_NAME-{micro/macro_doc}-{fuzzy-strict}`. This schema makes it easy to filter for a particular metric with `grep`.

The detailed json-report contains all the measures that are available (see below).

Moreover, the figures are computed in aggregate (Label = ALL) and per token-type.

#### Available figures and metrics
For any of the evaluation schemes described in [David Batista's blog post](http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/), the evaluation provides the following figures:

- `correct`
- `incorrect`
- `partial`
- `missed`
- `spurious`
- `possible`
- `actual`
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
We report baseline scores Named Entity Recognition and Classification (NERC) tasks trained with a basic CRF model. The model uses surface information only and dismisses the segmentation structure as it treats any particular document as a single, long sentence. The model is trained on the official training data and evaluated on the development set.

To train and evaluate a baseline model for the *NERC Coarse-grained* task, you can run the following command:

```python baseline.py --train TRAIN.tsv --dev DEV.tsv  --pred OUTPUT.tsv --cols NE-COARSE-LIT,NE-COARSE-METO --eval nerc_coarse```

`--cols` defines the columns for which baselines are trained.

`--eval` defines the task on which the baseline is evaluated.



The script outputs three files: a file with the predictions of the baseline system, a condensed report and a detailed report (see evaluation).

Please note that there is no baseline for named entity linking (NEL).

## Acknowledgments
Our evaluation module is based on David Batista's [NER-Evaluation](https://github.com/davidsbatista/NER-Evaluation).

