# AIDA-EE Dataset

The AIDA-EE Dataset contains 300 documents with 9,976 entity names linked to
Wikipedia (2010-08-17 dump). The documents themselves are taken from the APW
part of the GIGAWORD5 [1] dataset, with 150 documents from 2010-10-01
(development data) and 150 documents from 2010-11-01 (test data). Due to
licensing issues, we do not provide the document content, just the offsets
with the entity annotations.

The data was released as part of the Hoffart, Altun, and Weikum: Discovering
Emerging Entities with Ambiguous Names [2].


## Data Format

The dataset is split into two files: apw_eng_201010.tsv and
apw_eng_201011.tsv, corresponding to the (unzipped) apw_eng_201010 and
apw_ang_201011 files in the GIGAWORD5 dataset [1].

Each file contains one entity per line as tab-separated values, with the
following column meanings:

0: character offset in the original file (\n counts as character) - you need to (g)-unzip the original file for the offsets to make sense.
1: full string of the annotated name (recognized automatically by the Stanford CoreNLP toolkit [3], manually corrected)
2: YAGO2 identifier of the entity OR --OOKBE-- (out-of-knowledge-base entity, emerging entity)
3: Wikipedia URL (in 2010-08-17 dump) OR --OOKBE-- (out-of-knowledge-base entity, emerging entity)
4: Document ID (just for reference)


## Detailed Statistics

documents: 300
mentions: 9,976
mentions with emerging entities EE: 561
words per article (avg.): 538
mentions per article (avg.): 33 
entities per mention (avg using AIDA entity repository from 2010-08-17): 104
 

## Contact

The dataset was created at the Max Planck Institute for Informatics by
Johannes Hoffart and Gerhard Weikum, with the help of Dobromir Vasilev and Artem Galushko
as part of the AIDA project: http://www.mpi-inf.mpg.de/yago-naga/aida/

Contact eMail: <NAMEOFTHEPROJECTMENTIONINTHELINEABOVE>@mpi-inf.mpg.de


## FAQ

Q: Why are some mentions in the annotations file different than the ones in the original text?
A: This is due to the tokenization done by Stanford CoreNLP, where ' is split from the token. 
   This is the case for a few mentions, e.g. for Qatar Airways' in APW_ENG_20101101.0173. 
   However, the offsets are correct, just the mention we are using for annotation was
   a bit different. All problems are in the 201011 part of the annotations:
 - in line 121, original file offset 25594: annotation was Kurdistan Workers ' Party, orig says Kurdistan Workers' Party,
 - in line 3584, original file offset 330218: annotation was Qatar Airways , orig says Qatar Airways'
 - in line 3588, original file offset 330309: annotation was San ' , orig says San'a.
 - in line 4307, original file offset 411568: annotation was Kurdistan Workers ' Party , orig says Kurdistan Workers' Party h

## References 

[1] http://catalog.ldc.upenn.edu/LDC2011T07
[2] J. Hoffart, Y. Altun, and G. Weikum: Discovering Emerging Entities with Ambiguous Names. Proceedings of the 23rd International World Wide Web Conference, WWW 2014, Seoul, South Korea, 2014, pp. 385–395.
[3] http://nlp.stanford.edu/software/corenlp.shtml