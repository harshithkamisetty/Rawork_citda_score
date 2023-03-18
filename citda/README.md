# CITDA Scores
**CITDA** is an abbreviation for *Contextualized Interpretable Dictionary-based Text Analysis*.

## SHAP Values

SHAP values are a way to explain the output of a machine learning model. A SHAP value represents a feature's contribution to a change in the model output.

> *Level of analysis: word, sentence, document*

## LWIC Scores
LWIC scores are extracted from a dictionary of word categories. The dictionary is used to calculate the LWIC score for each word in a text. Each category has a score between 0 and 100.

> *Level of analysis: word, sentence (preferred)*

## CITDA Scores

CITDA scores are calculated by multiplying the LWIC score for each word in a text by the SHAP value for that word. The resulting scores are then summed to produce a single score for the text.

> *Level of analysis: word (preferred)*
