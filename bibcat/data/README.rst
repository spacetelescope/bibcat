Data directory
==============

This directory contains a script to build the ML input text data file. 
Due to the copyright license issues, we are not able to provide the original
full text data. We assume you have your own text file in JSON. 

To construct the input text JSON file, you need two sets of data: 
the MAST Papertrack DB data for paper classification (a label per bibcode) 
and the ADS full text data (a full text per bibcode). 
They both should be in the JSON format. 

The metadata keys for the MAST papertrack data, the ADS full text data, 
and the combined data are following though we only need ["abstract", "author", 
"bibcode", "body", "keyword", "keyword_norm", "pubdate", "title"] from 
the ADS text metadata to construct the final input data.

- keys_papertext:
  ['bibcode', 'abstract', 'author', 'bibstem', 'identifier', 'keyword', 
  'keyword_norm', 'page', 'pub', 'pub_raw', 'pubdate', 'title', 'volume', 
  'aff_canonical', 'institution', 'body']
- keys_papertrack:
  ['bibcode', 'searches':['search_key','ignored'], 'class_missions':['bibcode','papertype']]
- The combined input data:
  ['bibcode', 'abstract', 'author', 'keyword', 'keyword_norm', 'pubdate', 'title', 'body', 
  'class_missions', 'is_ignored_mast', 'is_ignored_hubble']

  For more information, you can contact the developers.
