__This is an updated version that improves the EL evaluation__

# CLEF-HIPE-2020-scorer

CLEF-HIPE-2020-scorer is a python module for evaluating Named Entity Recognition and Classification (NER) models and Entity Linking (EL) as defined in the [CLEF-HIPE-2020 Shared Task](https://impresso.github.io/CLEF-HIPE-2020//). The format of the data is similar to [CoNLL-U](https://universaldependencies.org/format.html), yet tokens may have more than one named entity annotations. Different annotations are recorded in different columns, which are evaluated separately (for more detail on the input format, refer to the [HIPE participation guidelines](https://zenodo.org/record/3604238)). 

The NERC evaluation goes beyond a token-based schema and considers entities as the unit of reference. An entity is defined as being of a particular type and having a span with a token onset and an offset. Following this definition, various NERC system outcomes may exist regarding correct entity types and boundaries when comparing the system response to the gold standard. The task of named entity evaluation is described in detail in a [blog post](http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/) by [David Batista](https://github.com/davidsbatista). Our scorer builds on the code in the [original repository](https://github.com/davidsbatista/NER-Evaluation), which accompanied the blog post.

### Metrics

#### NERC

NERC is evaluated in terms of **macro and micro Precision, Recall, F1-measure**. Two evaluation scenarios are considered: **strict** (exact boundary matching) and **fuzzy** (overlapping boundaries). 

Each column is evaluated independently, according to the following metrics:

- Micro average P, R, F1 at entity-level (not at token level), i.e. consideration of all true positives, false positives, and false negative over all documents. 
  - strict (exact boundary matching) and fuzzy (at least 1 token overlap).
  - separately per type and cumulative for all types.
  
- Document-level macro average P, R, F1 at entity-level (not on token level). i.e. average of separate micro scores across documents.
  - strict and fuzzy
  - separately per type and cumulative for all types
  

Our definition of macro differs from the usual one, and macro measures are computed as aggregates on document-level instead of entity-type level. Specifically, macro measures average the corresponding micro scores across all the documents, accounting  for (historical) variance in document length and entity distribution within documents instead of overall class imbalances.

Note that in the strict scenario, predicting wrong boundaries leads to severe punishment of one false negative (entity present in the gold standard but not predicted by the system) and one false positive (predicted entity by the system but not present in the gold standard). Although this may be severe, we keep this metric in line with [CoNLL](https://www.clips.uantwerpen.be/conll2000/chunking/output.html) and refer to the fuzzy scenario if the boundaries of an entity are considered as less important.

The Slot Error Rate (SER) is dropped for the shared task evaluation.

#### Entity Linking

The evaluation for EL works similarly to NERC. The link of an entity is interpreted as a label. ~~As there is no IOB-tagging, a consecutive row of identical links is considered as a single entity.~~ This version uses the named entities boundaries to determine the EL boundaries, however, it is still possible to use the original evaluation. In terms of boundaries, EL is evaluated according to the fuzzy scenario only. Thus, to get counted as correct, the system response needs only one overlapping link label with the gold standard. Literal and metonymic linking are evaluated separately.

EL strict regime considers only the system's top link prediction (NIL or QID), while the fuzzy regime expands system predictions with a set of historically related entity QIDs. For example, “Germany” QID is complemented with the QID of the more specific “Confederation of the Rhine” entity and both are considered as valid answers. The resource allowing for such historical normalization was compiled by the task organizers for the entities of the test data sets, and is released as part of the HIPE scorer. For the fuzzy regime, participants were invited to submit more than one link, and F-measure is additionally computed with cutoffs @3 and @5.

## Scorer

### Standard Evaluation

To evaluate the predictions of your system, you can run the following command:

```python clef_evaluation.py --ref GOLD.tsv --pred PREDICTIONS.tsv --task TASK --outdir RESULT_FOLDER```

There are three different evaluation modes (`TASK`) available: `nerc_coarse`, `nerc_fine`, `nel`. Depending on the task, the script performs the evaluation for the corresponding columns and evaluation scenarios automatically. 

Unless you define otherwise with the `--skip_check` argument, the evaluation script enforces the following convention on the file name (see [HIPE Shared Task Participation Guidelines](https://zenodo.org/record/3604238), p. 13): `TEAMNAME_TASKBUNDLEID_LANG_RUNNUMBER.tsv`

**Format requirements** The script expects both files (system responses and gold standard) to have a similar structure (same number of columns) as well as similar content (same number of token lines, in the exact same order). Any comment lines starting with a `#` may be omitted. The script will try to reconstruct the segmentation according to the gold standard automatically. In cases of unresolvable mismatches, the evaluation fails and outputs information about the issue. 

### Advanced Evaluation 

Moreover, the scorer allows for a detailed evaluation of performance on diachronic and noisy data for NERC and EL. 

To get evaluation results with a breakdown by noise-level,  use the argument `--noise-level`. The level of noise is defined as the length-normalized Levenshtein distance between the surface form of an entity and its human transcription. This distance is parsed from the column `MISC` of the gold standard per token (e.g., `LED0.0`). 

Example: `--noise-level 0.0-0.0,0.001-0.1,0.1-0.3,0.3-1.1` (lower bound <= LED < upper bound)

To get evaluation result with a breakdown by time periods,  use the argument `--time-period`.  The date is parsed from the document segmentation in the gold standard (e.g., `# document_id = NZZ-1798-01-20-a-p0002`) . 

Example: ` --time-period 1790-1810,1810-1830,1830-1850,1850-1870,1870-1890,1890-1910,1910-1930,1930-1950,1950-1970`  (lower bound <= date < upper bound)

For EL only, systems may provide a ranked list of links per entity. To evaluate at a particular cutoff, use the argument `--n_best` value(s). Per default are evaluated at a cutoff of 1 (top link only). An arbitrary number of links may be provided in the respective columns of the system response, separated by a pipe (`Q220|Q1747689|Q665037|NIL|Q209344`).

Example: `--n_best 1,3,5` (cutoffs @1, @3, @5)

If you provide more than one of these advanced evaluation options, all possible combinations will be computed. 



### Output

The evaluation script writes two files into the provided output folder:  `results_TASK_LANG.tsv` and `results_TASK_LANG_all.json`

The condensed tsv-report (`results_TASK_LANG.tsv`) contains all the measures that are relevant for the official shared task evaluation. The report has the following structure:

| System                               | Evaluation               | Label | P    | R    | F1   | F1_std | P_std | R_std | TP   | FP   | FN   |
| ------------------------------------ | ------------------------ | ----- | ---- | ---- | ---- | ------ | ----- | ----- | ---- | ---- | ---- |
| TEAMNAME_TASKBUNDLEID_LANG_RUNNUMBER | NE-FINE-COMP-micro-fuzzy | ALL   |      |      |      |        |       |       |      |      |      |

Cells may be empty in case they are not defined or provide only redundant information. The column `Evaluation` refers to the evaluated column and defines the measures P, R, F1, etc. It has the following structure: `COL_NAME-{micro/macro_doc}-{fuzzy-strict}`. This schema makes it easy to filter for a particular metric with `grep`.

The detailed json-report (`results_TASK_LANG_all.json`) contains all the measures that are available (see below).

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

We report baseline scores for Named Entity Recognition and Classification (NERC) using a trained basic CRF model. The model only uses surface information and dismisses the segmentation structure as it treats any particular document as a single, long sentence. The model is trained on the official training data and evaluated on the test set (except for English it is trained on the development set as there is no training data).

To train and evaluate a baseline model for the *NERC Coarse-grained* task, you can run the following command:

```python baseline.py --train TRAIN.tsv --dev TEST.tsv  --pred OUTPUT.tsv --cols NE-COARSE-LIT,NE-COARSE-METO --eval nerc_coarse```

`--cols` defines the columns for which baselines are trained.

`--eval` defines the task on which the baseline is evaluated.

The script outputs three files: a file with the predictions of the baseline system, a condensed report and a detailed report (see evaluation).

Please note that there is no script to produce the baseline for entity linking (EL).

## Acknowledgments
Our evaluation module is based on David Batista's [NER-Evaluation](https://github.com/davidsbatista/NER-Evaluation).

