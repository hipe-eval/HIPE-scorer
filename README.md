
# HIPE-scorer 

The HIPE-scorer is a python module for evaluating Named Entity Recognition and Classification (NER) models and Entity Linking (EL) as defined in the [CLEF-HIPE-2020](https://impresso.github.io/CLEF-HIPE-2020//) and [HIPE-2022](https://hipe-eval.github.io/HIPE-2022/) Shared Tasks. The format of the data is similar to [CoNLL-U](https://universaldependencies.org/format.html), yet tokens may have more than one named entity annotations. Different annotations are recorded in different columns, which are evaluated separately. For more detail on the input format, refer to the [CLEF-HIPE-2020](https://zenodo.org/record/3604238) and the [HIPE-2022](https://zenodo.org/record/6045662) participation guidelines. 

The NERC evaluation goes beyond a token-based schema and considers entities as the unit of reference. An entity is defined as being of a particular type and having a span with a token onset and an offset. Following this definition, various NERC system outcomes may exist regarding correct entity types and boundaries when comparing the system response to the gold standard. 


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


#### Entity Linking

EL evaluation is based on the same metrics as NERC and the link of an entity is interpreted as a label. However, the definition of strict and fuzzy regimes differs, according to what is relaxed (label or boundaries).

In the context of the HIPE Shared Tasks, we are interested in the capacity of systems to provide the correct link rather than the correct boundaries (something already evaluated in NERC).

The scorer provides the following evaluation regimes:    
1. **boundary-strict & label-strict**: the system response needs the correct link label on each token of the linked mention (an IOB mapping between NE mention and NE links is performed during evaluation and allows to check boundaries). This scenario is similar to what is done in GERBIL or NELEval. This scenario is **not used** during the HIPE shared tasks.
2. **boundary-fuzzy & label-strict**: at entity level, the system response needs only one entity token with a link label that overlaps with the gold standard entity link label to be counted as correct, and only the top link prediction (NIL or QID) is considered (cutoff @1). This scenario is used during the HIPE shared tasks.
3. **boundary-fuzzy & label-fuzzy**:  at entity level, the system response needs only one entity token with a link label that overlaps with the gold standard entity link label to be counted as correct, and more than one link is considered. For this regime,  participants are invited to submit more than one link (multiple QIDs can be specified, separated by `|` - please refer to guidelines specified above), and F-measure is additionally computed with cutoffs @3 and @5 . Furthermore, system predictions can be expanded with a set of historically related entity QIDs (by running first `normalize_linking.py`). For example, “Germany” QID is complemented with the QID of the more specific “Confederation of the Rhine” entity and both are considered as valid answers. The resource allowing for such historical normalization was compiled by the task organizers for the entities of the test data sets, and is released as part of the HIPE scorer. This scenario is used during the HIPE shared tasks.

In every scenario, literal and metonymic linking are evaluated separately.

:warning: **Important note - March 2022**
- EL evaluation regime (1) was introduced with this [PR](https://github.com/hipe-eval/HIPE-scorer/pull/17) by Adrián [creat89](https://github.com/creat89) (many thanks!) to correct issue [#13](https://github.com/hipe-eval/HIPE-scorer/issues/13) of the non-differentiation by the scorer of consecutive entity with NIL links (given that originally non IOB mapping was done). However, in the context of the HIPE shared tasks, the scorer should allow fuzzy boundaries (scenarios 2 and 3), while having issue [#13](https://github.com/hipe-eval/HIPE-scorer/issues/13) corrected. We are working on this. Presently, the `--task=nel` evaluates according to regime (1).
- EL evaluation regime (2) can be used by specifing `--task=nel --original_nel`. Presently this will consider consecutive NIL entities as one (will be fixed).
- EL evaluation regime (3) can be used by specifing `--n_best=<n>` (see usage of `clef_evaluation.py`)

For the 'EL-only' task where entity mentions are provided, regimes 1 and 2 will give the same results.

A new version will soon be published.

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

## Installation
```
$ python3 -mvenv venv
$ source venv/bin/activate
$ pip3 install -r requirements.txt
$ # for development
$ pip3 install -e .
```

## Release history and credits
- **v2.0**, **upcoming**: pre-HIPE2022 release, based on [Adrián creat89](https://github.com/creat89])'s [PR](https://github.com/hipe-eval/HIPE-scorer/pull/17) and a few additional corrections.
- **v1.1**, June 2020: post-HIPE-2020 evaluation release.
- **v1.0**, June 2020: version of the scorer as used during the CLEF-HIPE-2020 evaluation period.
- **v0.9**, Feb 2020,: first release of the HIPE scorer developed by Alex Flückiger, based on David Batista's [NER-Evaluation module](https://github.com/davidsbatista/NER-Evaluation) (see also his [blog post](http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/) on the topic).
