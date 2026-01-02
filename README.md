# A Mechanistic Explanation of the Effect of Existential Presuppositions on Hallucinations in Large Language Models

## D4EP Dataset
- Find the *D4EP* dataset in the `dataset` folder.

## Effect Quantification
- Find code and data for effect quantification in the `1_quantification` folder.
    - The `results` folder contains all human-annotated LLM responses.
    - The scripts `annotator.py` and `annotator_existence.py` are annotation tools for property questions and existence questions, respectively.
    - The script `runExperimentOllama.py` is used to generate the LLM responses.
    - The folder `openai` includes the code to generate batches for the openai api, and parse the batch responses to our json format (as all files in `results`). 

## Effect Explanation
- Find code for the effect explanation in the `2_explanation` folder.
    - The scripts `make_latent_data_*.py` extract latent representations from the respective LLM.
    - The script `eval_probes.py` is used to train and evaluate probes on the extracted latent data.
    - The notebook `test_rocs.ipynb` is used to test, whether ROC curves differ significantly.
 
# License
This work is licensed under a [Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
