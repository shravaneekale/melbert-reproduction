# Annotation guidelines: word-level metaphor (MIPVU)

These guidelines adapt **MIPVU** (Steen et al., 2010), the procedure used to annotate the VU Amsterdam Metaphor Corpus (VUA) that MelBERT is trained on. Following the same procedure means our new labels are comparable to MelBERT's training data.

## The task

You'll see one sentence at a time with one **target word** in `[brackets]`. Decide whether that word, **as used in this sentence**, is

* **M (metaphorical)**: a *metaphor-related word*, or
* **L (literal)**.

Only the bracketed word counts. Other words in the sentence may be metaphorical too, but you'll get those as separate items.

## The decision procedure

For the target word:

1. **Contextual meaning.** Read the whole sentence and work out what the word means *here*.
2. **Basic meaning.** Look the word up in a learner's dictionary: **Macmillan** (https://www.macmillandictionary.com, archived) or **Longman** (https://www.ldoceonline.com), the dictionaries MIPVU uses. Use the entry for the same part of speech. The basic meaning is the one that is
   * more **concrete** (you can see, hear, feel, smell or taste it),
   * related to **bodily action**,
   * more **precise** (less vague), or
   * historically **older**.

   The basic meaning is not necessarily the most frequent one.
3. **Contrast + comparison.** If the contextual meaning is **different** from the basic meaning **and** can be understood **by comparison with it**, label **M**. Otherwise label **L**.

If the word has only one sense in the dictionary, or the contextual sense *is* the basic sense, label **L**.

## Rules of thumb

| Situation | Label | Example |
|---|---|---|
| Conventional (dead) metaphor, still with a more basic sense | **M** | prices *climbed*; she *grasped* the idea; a *huge* improvement |
| Abstract use of a spatial preposition | not a target (function word) | |
| Personification of a device | **M** | my laptop *crawls*; the update *killed* the battery |
| Hyperbole that is literally the same sense | **L** | it took *forever* (time sense) |
| Technical term whose technical sense is now the basic one | usually **L** | a neural *network*, a *tree* data structure (check the dictionary; if the technical sense is listed separately and the concrete sense is more basic, label **M**) |
| Phrasal verb or multi-word expression treated as a unit | label the bracketed part by MIPVU's rule: if the whole unit is listed in the dictionary as one lexical unit, judge the unit | |
| Simile / explicit comparison (*like a rock*) | **M** for the compared word (MIPVU "direct metaphor") | |
| Typo or unclear word | your best guess, plus mark low confidence | |

## Procedure rules

* **Work independently.** Don't look at the other annotator's labels, MelBERT's predictions or the LLM's labels before you finish.
* Mark **low confidence** (`m?` / `l?` in the terminal tool, or `x` in the spreadsheet) when you are unsure. This doesn't change the label; it's only used in analysis.
* Use the **note** field for anything surprising (e.g. "basic sense of *drive* = control a vehicle").
* Take a break every ~150 items. Fatigue lowers agreement.

## Agreement and adjudication

* A1 (primary annotator) labels **every** item.
* A2 independently labels the **double-annotated subset** (30% of sentences, stratified by domain).
* We report Cohen's κ (overall, per domain, per POS). For comparison, the VUAMC reliability tests report Fleiss' κ of about 0.85 between four trained annotators (Steen et al., 2010). Non-expert annotators typically reach lower agreement (κ ≈ 0.65–0.72 when MIPVU has been applied to other languages).
* Disagreements are listed in `data/annotation/adjudication.csv`. Both annotators discuss each one, re-check the dictionary, and record the final label and a short reason. The adjudicated label becomes gold.

## Reference

Steen, G. J., Dorst, A. G., Herrmann, J. B., Kaal, A. A., Krennmayr, T., & Pasma, T. (2010). *A Method for Linguistic Metaphor Identification: From MIP to MIPVU.* John Benjamins.
