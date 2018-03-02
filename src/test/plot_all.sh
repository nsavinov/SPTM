#!/usr/bin/env bash
rm params.out
rm tmp.out
for file in $@;
do 
  cat $file >> tmp.out
  
  echo "Log file: $file" >> params.out
  echo "" >> params.out
  head -20 $file >> params.out
  echo "-------" >> params.out
  echo "" >> params.out
done
  
python2 process_log.py tmp.out > table_results.txt
rm tmp.out
