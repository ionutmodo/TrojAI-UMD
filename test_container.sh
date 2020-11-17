#!/bin/bash

sudo singularity run -B /home/ubuntu/workplace/TrojAI-data/id-0000$2/ $1 --model_filepath /home/ubuntu/workplace/TrojAI-data/id-0000$2/model.pt --result_filepath /home/ubuntu/workplace/TrojAI-data/id-0000$2/result.txt --scratch_dirpath /home/ubuntu/workplace/TrojAI-data/id-0000$2/scratch --examples_dirpath /home/ubuntu/workplace/TrojAI-data/id-0000$2/clean_example_data
