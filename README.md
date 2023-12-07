> [!NOTE]
> This is an extension of the speech classifier program developed by [Thomas Davidson et al.](https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master). As Thomas Davidson's repository is no longer maintained, we decided to create our own, added modifications, and tested it with different datasets, including speech data from [Berkeley](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech)


There are 6 files in the "speech_classifier" folder:

1. generate_group_csv.py
1. count_groups.py
1. generate_trained_model.ipynb
1. speech_classifier.py
1. generate_cv_data.py
1. run_cross_validation.py
* The "data" folder referenced below is available in [data.zip  v1.0.0](https://github.com/MoSadri/Thesis_2023/releases/download/v1.0.0/data.zip)
* Note that the files in the input folder are aligned with the original analysis data set of TDavidson (i.e., labeled_data.csv). In addition, the data entries are labeled with target groups in the files in the input folder.


All .py files can be executed with a simple command like this:

``` python program_name.py ``` 

The first program: **generate_group_csv.py** reads the speech data available in the "data" folder (provided by Berkeley) and selects the desired number of groups to be analyzed. Different scenarios can be created for testing by altering the number of each targeted group in the file according to the "data_name" variable specified. For simplicity in our description, we will use the "balanced" data name as an example.

- Input: ../data/berkeley_speech_dataset.csv
- Output: ../data/balanced_dataset.csv

The second program: **count_groups.py** is a simple script to print out the actual number of groups resulting from the first program. <br>
As there are overlaps between different targeted groups, it is useful to obtain the percentage of each targeted group produced.

- Input: ../data/balanced_dataset.csv
- Output: ../output/balanced_groups_counts.txt

The third program: **generate_trained_model.ipynb** is a Jupyter Notebook script to generate the trained model that needs to be passed into the speech classifier program. <br> This program will produce 5 files with a .pkl extension, which need to be passed to speech_classifier.py.

- Input: ../data/balanced_dataset.csv
- Output:

1. ../data/balanced_model.pkl
2. ../data/balanced_tfidf.pkl
3. ../data/balanced_idf.pkl
4. ../data/balanced_pos.pkl
5. ../data/balanced_oth.pkl

The file: **speech_classifier.py** is the actual speech classifier program. It analyzes all the speech files in the input folder and produces the number of hate speech instances detected in the input files. It is currently also set to analyze a pre-labeled data file named "labeled_data.csv."  <br> This file was created by TDavidson to test the program's performance. We use this same file to determine the accuracy, precision, recall, and F1 score of our program.

Trained model files:

1. ../data/balanced_model.pkl
1. ../data/balanced_tfidf.pkl
1. ../data/balanced_idf.pkl
1. ../data/balanced_pos.pkl
1. ../data/balanced_oth.pkl


- Input CSV files to be analyzed: all files located in ../input
- Output: The output is located in ../output. The program will produce one output file for every file it finds in the input directory, listing the predicted class for each tweet/text within each file.
- It will also generate two PDF files and two TXT files. The PDF files are named "original_hate_vs_balanced_hate.pdf" and "original_hate+offensive_vs_balanced_hate.pdf" and contain confusion matrices based on the analysis of the original analysis data set "labeled_data.csv." The TXT files are named "original_hate_vs_balanced_hate.txt" and "original_hate+offensive_vs_balanced_hate.txt" and contain quality scores of the classifier program based on the analysis of the input CSV files.
  
<br> We concentrate on just two classes, "hate" or "not hate" in our program (and designed our trained dataset accordingly). Therefore, we produce the file "original_hate_vs_balanced_hate.pdf" by considering all "Offensive" class instances in "labeled_data.csv" as incorrectly classified. The second file, "original_hate+offensive_vs_balanced_hate.pdf," treats all "Hate" and "Offensive" class instances the same as "Hate," resulting in higher accuracy.

**Cross Validation**:
There are two files
1. generate_cv_data.py and run_cross_validation.py. 
- These two files are used for performing k-fold cross-validation. 
- Currently, k is set to 5, but users can set it to different numbers to perform k-fold cross-validation.

The steps are:
1. python generate_group_csv.py 
1. python generate_cv_data.py
1. python run_cross_validation.py

- **The first program**: generate_group_csv.py is used to produce training data with the desired number for each targeted group. This is the same program as before.
  - Input: ../data/berkeley_speech_dataset.csv
  - Output: ../data/balanced_dataset.csv
 

- **The second program**: generate_cv_data.py is used to split the files into k+1 pieces, in preparation for k-fold validation. With one piece used as analysis dataset, and 5 other pieces used as training datasets. This step will be repeated k times so each piece of data will have its chance to be the analysis dataset. This program will also generate the .pkl files needed for each trained dataset. <br>
  - Input: ../data/balanced_dataset.csv
  - Output: 
Analysis sets: ../cv_data/balanced_cvanalysis_fold1.csv ../cv_data/balanced_cvanalysis_fold2.csv ... ../cv_data/balanced_cvanalysis_foldk.csv <br>
Training sets: balanced_cvtrain_fold1.csv balanced_cvtrain_fold2.csv ... balanced_cvtrain_foldk.csv <br>
Pkl files: balanced_cvtrain_fold1_idf.pkl balanced_cvtrain_fold1_model.pkl balanced_cvtrain_fold1_oth.pkl balanced_cvtrain_fold1_pos.pkl balanced_cvtrain_fold1_tfidf.pkl ... balanced_cvtrain_foldk_tfidf.pkl <br>


- Program for running the actual cross-validation: <br>
run_cross_validation.py will run through all the analysis sets and training sets for each fold and generate quality scores for each file. These quality scores include accuracy, precision, recall, and F1 score.

- Input: "Analysis sets" and "Training sets" generated by generate_cv_data.py
- Output: ../cv_output/balanced_cvresults_fold1.txt, ../cv_output/balanced_cvresults_fold2.txt, ... ../cv_output/balanced_cvresults_foldk.txt


---------------------------------------------------------------------------------------------------------------------------------------------------------------------
> [!IMPORTANT]
> **Data Used**
> - Our program uses the newest Python version (3.11 at the time of our testing), which is an update from version 2.7 in the original TDavidson run. We obtained the trained data files (the .pkl files) from TDavidson's repository, repickled so they can be used in our program. These files are prefixed with "original_" and located in the [data folder](https://github.com/MoSadri/Thesis_2023/releases/download/v1.0.0/data.zip). Unfortunately, these files are trained models, and the CSV files used for generating these files aren't available, so it is not possible to modify these files.

TDavidson also uses an analysis set named "labeled_data.csv," which is a set of tweets with manually labeled classes ("Hate," "Offensive," or "Neither").

> [!IMPORTANT]
> In order to test with different data, we downloaded a new set of data from Berkeley researchers ([link](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech)), and name it "berkeley_speech_dataset.csv", which is put into the data folder as well. 



We use the "berkeley_speech_dataset.csv" to create three different scenarios to observe the effectiveness of this speech classifier program.
1. Scenario 1: csv files with tweets targeting mostly black
1. Scenario 2: csv files with tweets targeting mostly women
1. Scenario 3: csv files with tweets targeting a balanced group (including black, women and LGBT group)

> [!NOTE]
> Different scenarios can be created by setting different numbers when running the program "generate_group_csv.py". Here are our configurations:

| Scenario                  | Configuration                |
| :------------------------  | :--------------------------- |
| Configuration for scenario 1 | 9000 black, 500 women, 200 trans, 150 gay, 150 lesbian |
| Configuration for scenario 2 | 9000 women, 500 black, 200 trans, 150 gay, 150 lesbian |
| Configuration for scenario 3 | 3300 black, 3300 women, 2800 trans, 100 gay, 500 lesbian |

> [!NOTE]
> **Results**


| Scenario   | Target Group | Accuracy | Precision (Hate) | Recall (Hate) | F1 Score (Hate) |
|------------|--------------|----------|-------------------|---------------|-----------------|
| Black      | Black        | 67%        | 91%             | 65%         | 76%             |
| Black      | Women        | 74%        | 94%           | 71%         | 81%             |
| Black      | LGBT         | 84%        | 95%             | 86%               | 90%             |
| Women      | Black        | 62%      | **96%**               | 55%          | 70%             |
| Women      | Women        | 75%      | **96%**               | 71%        | 81%             |
| Women      | LGBT         | 70%      | **96%**               | 68%         | 76%            |
| Balanced   | Black        | **68%**        | 94%             | 64%          | 76%            |
| Balanced   | Women        | **83%**        | 94%              |84%          | 89%           |
| Balanced   | LGBT         | **85%**        | 95%              | 88%          | 91%          |

