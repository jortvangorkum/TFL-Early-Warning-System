# Early Warning System

To run the classifier, run the main in the classification folder. Please make sure the data-splitted folder is unzipped.
In order to change the importance of the Fail class, change the input of the create_trees() function to another list with the desired values.

### Results

The results are located in data\results, which includes:

- Figures with the scores of the different decision trees, per fail-class-importance and number of months of assessments.
- A csv file containing all scores.

### Explanations

The explanations of specific samples can be printed. To select a sample, give a fail-class-importance, the months of assessments that are
taken into account, and the index in the test set to the function explain_sample(). The explanations will be printed in the console.

## Data analysis

```
jupyter nbconvert analysis.ipynb --to pdf --output analysis.pdf
```