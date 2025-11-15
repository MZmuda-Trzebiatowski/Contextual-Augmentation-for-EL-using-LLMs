|---------------------|
| KORE50 AIDA Dataset |
|---------------------|

2012-07-18
Johannes Hoffart

Overview
--------

This dataset was created for [1] as a difficult dataset to test different semantic relatedness measures as part of entity disambiguation with AIDA [2]. It contains 50 sentences from various domains (politics, sports, celebrities, music, and business) that were handcrafted to be difficult for entity disambiguation. See [1] for more details.
 
File Format
-----------

The format of AIDA.tsv is the following:

- Each document starts with a line: -DOCSTART- (<docid>)
- Each following line represents a single token, sentences are separated by an empty line
  
Lines with tabs are tokens the are part of a mention:
- column 1 is the token
- column 2 is either B (beginning of a mention) or I (continuation of a mention)
- column 3 is the full mention used to find entity candidates
- column 4 is the corresponding YAGO2 [3] entity (in YAGO encoding, i.e. unicode characters are backslash encoded and spaces are replaced by underscores, see also the tools on the YAGO2 website), OR --NME--, denoting that there is no matching entity in YAGO2 for this particular mention, or that we are missing the connection between the mention string and the YAGO2 entity.


License
-------

The KORE50 dataset is licensed under the Creative Commons Attribution 3.0 License (http://creativecommons.org/licenses/by/3.0/)


References
----------

[1] J. Hoffart, S. Seufert, D. B. Nguyen, M. Theobald, and G. Weikum, “KORE: Keyphrase Overlap Relatedness for Entity Disambiguation,” presented at the Proceedings of the 21set ACM International Conference on Information and Knowledge Management, CIKM 2012, Hawaii, USA, 2012.
[2] J. Hoffart, M. A. Yosef, I. Bordino, H. Fürstenau, M. Pinkal, M. Spaniol, B. Taneva, S. Thater, and G. Weikum, “Robust Disambiguation of Named Entities in Text,” presented at the Conference on Empirical Methods in Natural Language Processing, EMNLP 2011, Edinburgh, 2011, 2011, pp. 782–792.
[3] J. Hoffart, F. M. Suchanek, K. Berberich, and G. Weikum, “YAGO2: A spatially and temporally enhanced knowledge base from Wikipedia,” Artificial Intelligence, Jun. 2012.