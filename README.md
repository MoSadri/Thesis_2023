This is an extension of the speech classifier program developed by Thomas Davidson et al. (https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master). As Thomas Davidson's repository is no longer maintained, we decided to create our own, added modifications, and tested it with different datasets, including speech data from Berkeley (https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech)

This site was built using [GitHub Pages](https://pages.github.com/).


There are 6 files in the "speech_classifier" folder:

* generate_group_csv.py
* count_groups.py
* generate_trained_model.ipynb
* speech_classifier.py
* generate_cv_data.py
* run_cross_validation.py


All .py files can be executed with a simple command like this:

``` python program_name.py ``` 

The first program: **generate_group_csv.py** reads the speech data available in the "data" folder (provided by Berkeley) and selects the desired number of groups to be analyzed. Different scenarios can be created for testing by altering the number of each targeted group in the file according to the "data_name" variable specified. For simplicity in our description, we will use the "balanced" data name as an example.

- Input: ../data/berkeley_speech_dataset.csv
- Output: ../data/balanced_dataset.csv

The second program: **count_groups.py** is a simple script to print out the actual number of groups resulting from the first program. As there are overlaps between different targeted groups, it is useful to obtain the percentage of each targeted group produced.

Input: ../data/balanced_dataset.csv
Output: ../output/balanced_groups_counts.txt

The third program: **generate_trained_model.ipynb** is a Jupyter Notebook script to generate the trained model that needs to be passed into the speech classifier program. This program will produce 5 files with a .pkl extension, which need to be passed to speech_classifier.py.

Input: ../data/balanced_dataset.csv
Output:

1. ../data/balanced_model.pkl
2. ../data/balanced_tfidf.pkl
3. ../data/balanced_idf.pkl
4. ../data/balanced_pos.pkl
5. ../data/balanced_oth.pkl

The file: **speech_classifier.py** is the actual speech classifier program. It analyzes all the speech files in the input folder and produces the number of hate speech instances detected in the input files. It is currently also set to analyze a pre-labeled data file named "labeled_data.csv." This file was created by TDavidson to test the program's performance. We use this same file to determine the accuracy, precision, recall, and F1 score of our program.

Trained model files:

1. ../data/balanced_model.pkl
2. ../data/balanced_tfidf.pkl
3. ../data/balanced_idf.pkl
4. ../data/balanced_pos.pkl
5. ../data/balanced_oth.pkl


**Input CSV files to be analyzed**: all files located in ../input
**Output**: The program will produce one output file for every file it finds in the input directory, listing the predicted class for each tweet/text within each file. It will also generate two PDF files and two TXT files. The PDF files are named "original_hate_vs_balanced_hate.pdf" and "original_hate+offensive_vs_balanced_hate.pdf" and contain confusion matrices when analyzing "labeled_data.csv." 

The TXT files are named "original_hate_vsbalanced_hate.txt" and "original_hate+offensive_vs_balanced_hate.txt" and contain quality scores of the classifier program. We concentrate on just two classes, "hate" or "not hate" in our program (and designed our trained dataset accordingly). Therefore, we produce the file "original_hate_vs_balanced_hate.pdf" by considering all "Offensive" class instances in "labeled_data.csv" as incorrectly classified. The second file, "original_hate+offensive_vs_balanced_hate.pdf," treats all "Hate" and "Offensive" class instances the same as "Hate," resulting in higher accuracy.

**Cross Validation**:
There are two files: generate_cv_data.py and run_cross_validation.py. 
These two files are used for performing k-fold cross-validation. 
Currently, k is set to 5, but users can set it to different numbers to perform k-fold cross-validation.

The steps are:
python generate_group_csv.py 
python generate_cv_data.py
python run_cross_validation.py

Input: ../data/berkeley_speech_dataset.csv
Output: ../data/balanced_dataset.csv


**Analysis sets**: ../cv_data/balanced_cvanalysis_fold1.csv, ../cv_data/balanced_cvanalysis_fold2.csv, ... ../cv_data/balanced_cvanalysis_foldk.csv
**Training sets**: balanced_cvtrain_fold1.csv, balanced_cvtrain_fold2.csv, ... balanced_cvtrain_foldk.csv
**Pkl files**: balanced_cvtrain_fold1_idf.pkl, balanced_cvtrain_fold1_model.pkl, balanced_cvtrain_fold1_oth.pkl, balanced_cvtrain_fold1_pos.pkl, balanced_cvtrain_fold1_tfidf.pkl, ... balanced_cvtrain_foldk_tfidf.pkl

**Program for running the actual cross-validation**: run_cross_validation.py will run through all the analysis sets and training sets for each fold and generate quality scores for each file. These quality scores include accuracy, precision, recall, and F1 score.

Input: "Analysis sets" and "Training sets" generated from above
Output: ../cv_output/balanced_cvresults_fold1.txt, ../cv_output/balanced_cvresults_fold2.txt, ... ../cv_output/balanced_cvresults_foldk.txt


---------------------------------------------------------------------------------------------------------------------------------------------------------------------
> [!IMPORTANT]
> **Data Used**
> - Our program uses the newest Python version (3.11 at the time of our testing), which is an update from version 2.7 in the original TDavidson run. We obtained the trained data files (the .pkl files) from TDavidson's repository, repickled so they can be used in our program. These files are prefixed with "original_" and located in the > data folder. Unfortunately, these files are trained models, and the CSV files used for generating these files aren't available, so it is not possible to modify these files.

TDavidson also uses an analysis set named "labeled_data.csv," which is a set of tweets with manually labeled classes ("Hate," "Offensive," or "Neither").

> [!IMPORTANT]
> In order to test with different data, we downloaded a new set of data from Berkeley researchers ([link](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech)), and name it "berkeley_speech_dataset.csv", which is put into the data folder as well. 



We use the "berkeley_speech_dataset.csv" to create various three different scenarios to observe the effectiveness of this speech classifier program.
1. Scenario 1: csv files with tweets targeting mostly black
1. Scenario 2: csv files with tweets targeting mostly women
1. Scenario 3: csv files with tweets targeting a balanced group (including black, women and LGBT group)

> [!NOTE]
> Different scenarios can be created by setting different numbers when running the program "generate_group_csv.py". Here are our configuration:
> - Configuration for scenario 1: 9000 black, 500 women, 200 trans, 150 gay, 150 lesbian
> - Configuration for scenario 2: 9000 women, 500 black, 200 trans, 150 gay, 150 lesbian
> - Configuration for scenario 3: 3300 black, 3300 women, 2800 trans, 100 gay, 500 lesbian






