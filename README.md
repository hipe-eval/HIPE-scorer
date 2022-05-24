# HIPE-scorer 

The HIPE-scorer is a python module for **evaluating Named Entity Recognition and Classification (NER) and Entity Linking (EL) systems**.    

It has been developed and used in the context of the **HIPE ('Identifying Historical People, Places and other Entities')** shared tasks on NE processing on **historical documents**, with two evaluation campaigns:    



|   | Website  |  Data | Evalution Toolkit  |  Results |
|---|---|---|---|---|
| **HIPE-2022**  |  [HIPE-2022](https://hipe-eval.github.io/HIPE-2022/) | [HIPE-2022-data](https://github.com/hipe-eval/HIPE-2022-data/blob/main/README.md)  |  [HIPE-2022-eval](https://github.com/hipe-eval/HIPE-2022-eval) | [HIPE 2022 results]()  |
|  **CLEF-HIPE-2020** |  [CLEF-HIPE-2020](https://impresso.github.io/CLEF-HIPE-2020/) | [CLEF-HIPE-2020](https://github.com/impresso/CLEF-HIPE-2020/tree/master/data)  | [CLEF-HIPE-2020-eval](https://github.com/impresso/CLEF-HIPE-2020-eval)  | [CLEF HIPE 2020 results](https://github.com/impresso/CLEF-HIPE-2020/blob/master/evaluation-results/ranking_summary_final.md)  |

#### Release history
- 23 May 2020: [v2.0](), scorer as used during the HIPE-2022 evaluation (May 2022).
- 05 Jun 2020: [v1.1](https://github.com/hipe-eval/HIPE-scorer/releases/tag/1.1), post-HIPE-2020 evaluation release.
- 03 Jun 2020: [v1.0](https://github.com/hipe-eval/HIPE-scorer/releases/tag/1.0), scorer as used during the CLEF-HIPE-2020 evaluation (May 2020).
- 20 Feb 2020: [v0.9](https://github.com/hipe-eval/HIPE-scorer/releases/tag/v0.9), first release.


[Main functionalities](#main-functionalities)    
[Installation](#installation)  
[CLI usage](#cli-usage)  
[Forthcoming](Forthcoming)    
[License](license)

## Main functionalities

The scorer evaluates at the entity level, whereby entities (most often multi-words) are considered as the reference units, with a specific type as well as a token-based onset and offset. In the case of EL, the reference ID of an entity (or link) is considered as the label.

### Metrics

For both NERC and EL, the scorer compute the following metrics:  
   
- Micro average Precision, Recall, and F1-measure, based on true positives, false positives, and false negative figures computed over all documents.    
- Document-level macro average P, R and F1, based on the average of separate micro scores across documents.    

Please note that our definition of the macro sceheme differs from the usual one: macro measures are computed as aggregates at document-level and not at entity-type level. Specifically, the macro measures average the corresponding micro scores across all documents. This allow to account for variance in (historical) document length and entity distribution within documents, instead of overall class imbalances.

Measures are calculated separately by entity type, and cumulatively for all types.

### Evaluation regimes

There are different evaluation regimes depending on how strictly entity type and boundaries correctness is judged. The scorer provides strict and fuzzy evaluation regimes for both NERC and EL, as follows:

#### NERC
- **strict**: requires exact match of both entity type and entity boundaries.
- **fuzzy**: requires exact match of entity type and at least one token overlap.

#### Entity Linking
- **strict**: requires exact match of both entity link and entity boundaries. In other words, the system response needs the correct link label on each token of the linked mention (an IOB mapping between NE mention and NE links is performed during evaluation and allows to check boundaries). 
This setting is never used in HIPE shared tasks since we are interested in the systems ability to provide the correct link rather than the correct boundaries (something already evaluated in NERC).
- **fuzzy**: requires exact match of entity link and at least one token overlap. In other words, the system response needs only one entity token with the correct link label to be counted as correct. This is the default EL evaluation regime in HIPE shared tasks.
- **relaxed**: same are **fuzzy** above, with an additional flexibility at the link level. System predictions are expanded with a set of historically related entity QIDs. For example, “Germany” QID is complemented with the QID of the more specific “Confederation of the Rhine” entity and both are considered as valid answers. The resource allowing for such historical normalization was compiled by the HIPE team for both shared task editions. See below usage instructions.

For both EL *fuzzy* and *relaxed* setting, the number of link predictions taken into account can be adapted, i.e. system can provide multiple links or QIDs (separated by `|` ). The scorer can evaluate with cutoffs @1, @3 and @5.

## Installation

The scorer requires python 3 and and the module itself needs to be installed as an editable dependency: 

```
$ python3 -mvenv venv
$ source venv/bin/activate
$ pip3 install -r requirements.txt
$ # for development
$ pip3 install -e .
```


## CLI Usage

**Input data format** is similar to CoNLL-U, with multiple columns recording different annotations (when appropriate or needed) per entity tokens. Supported tagging schemes are IOB and IOBES.

Below is an example, see also the [CLEF-HIPE-2020](https://zenodo.org/record/3604238) and the [HIPE-2022](https://zenodo.org/record/6045662) participation guidelines for more details.


```
TOKEN	NE-COARSE-LIT	NE-COARSE-METO	NE-FINE-LIT	NE-FINE-METO	NE-FINE-COMP	NE-NESTED	NEL-LIT	NEL-METO	MISC
# hipe2022:document_id = NZZ-1798-01-20-a-p0002
# hipe2022:date = 1798-01-20
# ...
berichtet	O	O	O	O	O	O	_	_	_
der	O	O	O	O	O	O	_	_	_
General	B-pers	O	B-pers.ind	O	B-comp.title	O	Q321765	_	_
Hutchinson	I-pers	O	I-pers.ind	O	B-comp.name	O	Q321765	_	EndOfLine
—	O	O	O	O	O	O	_	_	_
```


### Standard evaluation

To evaluate the predictions of your system, run the following command:

```python clef_evaluation.py --ref GOLD.tsv --pred PREDICTIONS.tsv --task TASK --outdir RESULT_FOLDER```

Main parameters are (`clef_evaluation.py -h` to see full description):

-  `--task`: can take  `nerc_coarse`, `nerc_fine` or `nel` as value. Depending on the task, the script performs the evaluation for the corresponding columns and evaluation scenarios automatically. 
-   `--hipe_edition`: can take `hipe-2020` or `hipe-2022` as value [default  `hipe-2020`]. This impacts which columns are evaluated for each task, and which system response file naming convention is required. 
-   `--n_best=<n>`: to be used with `nel` task, specifies the cutoff value when provided with a ranked list of entity links [default: 1].
-  `--original_nel`: to be used with `nel` task, triggers the HIPE-2020 EL boundary splitting (with different NIL entities considered as one). 
- `--skip-check`:  skips the check that ensures that system response files name is in line with submission requirements (`TEAMNAME_TASKBUNDLEID_LANG_RUNNUMBER.tsv` for HIPE-2020 and `TEAMNAME_TASKBUNDLEID_DATASETALIAS_LANG_RUNNUMBER.tsv` for HIPE-2022).

**Format requirements** The script expects both system response and gold standard files to have a similar structure (same number of columns) as well as similar content (same number of token lines, in the exact same order). Any comment lines starting with a `#` may be omitted. The script will try to reconstruct the segmentation according to the gold standard automatically. In cases of unresolvable mismatches, the evaluation fails and outputs information about the issue. 

### Advanced Evaluation 

The scorer allows for a detailed evaluation of performance on diachronic and noisy data for NERC and EL. 

- To get evaluation results with a breakdown by noise-level,  use the argument `--noise-level`. The level of noise is defined as the length-normalized Levenshtein distance between the surface form of an entity and its human transcription. This distance is parsed from the column `MISC` of the gold standard per token (e.g., `LED0.0`). 

    Example: `--noise-level 0.0-0.0,0.001-0.1,0.1-0.3,0.3-1.1` (lower bound <= LED < upper bound)

- To get evaluation result with a breakdown by time periods,  use the argument `--time-period`.  The date is parsed from the document segmentation in the gold standard (e.g., `# document_id = NZZ-1798-01-20-a-p0002`) . 

    Example: ` --time-period 1790-1810,1810-1830,1830-1850,1850-1870,1870-1890,1890-1910,1910-1930,1930-1950,1950-1970`  (lower bound <= date < upper bound)

- For EL, to get the relaxed evaluation, run the script `normalize_linking.py` first. Provided with a link mapping, this script expand system prediction with historically-related QIDS. Setting used on HIPE 2020 and 2022.

If you provide more than one of these advanced evaluation options, all possible combinations will be computed. 

### Output

The evaluation script outputs two files in the provided output folder:     

- A condensed `results_TASK_LANG.tsv` report that contains the main relevant measures, with the following structure:

| System                               | Evaluation               | Label | P    | R    | F1   | F1_std | P_std | R_std | TP   | FP   | FN   |
| ------------------------------------ | ------------------------ | ----- | ---- | ---- | ---- | ------ | ----- | ----- | ---- | ---- | ---- |
| TEAMNAME_TASKBUNDLEID_LANG_RUNNUMBER | NE-FINE-COMP-micro-fuzzy | ALL   |      |      |      |        |       |       |      |      |      |

Cells may be empty in case they are not defined or provide only redundant information. The column `Evaluation` refers to the evaluated column and defines the measures P, R, F1, etc. It has the following structure: `COL_NAME-{micro/macro_doc}-{fuzzy-strict}`. This schema makes it easy to filter for a particular metric with `grep`.

 - A detailed json-report (`results_TASK_LANG_all.json`) that contains all measures and figures for each evaluation regimes, i.e.: 
 	- `correct`, `incorrect`, `partial`, `missed`, `spurious`    
	- `possible` (=number of annotations in the gold standard), `actual` (=number of annotations predicted by the system)    
	- `TP`, `FP`, `FN`    
	- `P_micro`, `R_micro`, `F1_micro`    
	- `P_macro_doc`, `R_macro_doc`, `F1_macro_doc`    
	- `P_macro_doc_std`,  `R_macro_doc_std`, `F1_macro_doc_std`    
	- `P_macro`, `R_macro`, `F1_macro`    
	- `F1_macro (recomputed from P & R)`    

	
Evaluation regimes (according to the script's internal naming):    

- **strict**: inner regime that orresponds to the *strict* evaluation.    
- **ent_type**: inner regime that corresponds to the *fuzzy* evaluation.    
- **partial**: inner regime that does not correspond to an 'public' HIPE evaluation scenario. The counter could be used for a very fuzzy evaluation regime, where a system entity is correct as long as there is a boundary overlap with a reference entity (i.e., type can be identical or not).     
- **exact**: inner regime that does not correspond to an 'public' HIPE evaluation scenario and focuses exclusively on boundaries (prediction considered as correct as long as boundaries are exact).    


## Forthcoming:    

- pip install
- read the doc
- scorer call via a function

## Contributors

The very first version of the HIPE scorer was inspired from David Batista's [NER-Evaluation module](https://github.com/davidsbatista/NER-Evaluation) (see also this [blog post](http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/)).

- Alex Fluckiger (first version for CLEF-HIPE-2020)
- Simon Clematide (maintainance and developement for HIPE 2022)
- Maud Ehrmann (maintainance and developement for HIPE 2022)
- Matteo Romanello (maintainance and developement for HIPE 2022)
- [Adrián creat89](https://github.com/creat89]) fixed the problem of consecutive NIL boundaries in this [PR](https://github.com/hipe-eval/HIPE-scorer/pull/17)


## License

The HIPE-scorer is licensed under the MIT License - see the [license](https://github.com/hipe-eval/HIPE-scorer/blob/master/LICENSE) file for details.
