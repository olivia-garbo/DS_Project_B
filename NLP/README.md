This project presents an end-to-end Natural Language Processing (NLP) pipeline designed to automatically extract character entities and family/social relationships from English novels.
The full workflow is implemented using spaCy, rule-based patterns, knowledge base linking, co-reference resolution, and dependency parsing.

The text of Pride and Prejudice is downloaded from Gutenburg named as 42671.txt
After cleaning the data (the code for data cleaning can be seen in pre_process.py), we can obtain cleaned text from clean_book.txt, which is ready token-sequence extrction method.(Main 1)

The charaters.csv displays the manually compiled charaters Knowledge base.

Main1.py illustrated Entity1-Relationship-Entity2 pattern extraction, which has been elaborated in in-progress report. 

The output of main1.py is conslidated_relationships.csv, which is also analysed in in-progress report.

Main2.py was compiled after in-progress for improvenment. It is a initial idea for establishing dependency parsing mothod without using co-reference resolution.

And charaters_updated.csv is reconstructed Knowledge base used for fianl report.

You can find resolved_book.txt, which is the content of novel after co-reference resolution.

After that, main3.py was created, which is conmbined by the method in main1 and main2.

The output of main3.py is consolidated_relationships.csv, which is explained in final report

Then you can run post_process_updated.py, the code inside are used to genrate relationship_with_counts.csv and relationship_pivot_summary.csv, and social network diagram. This code converted unstructual data to structural data for generating diagram.



