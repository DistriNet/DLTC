#!/bin/bash

curl 'https://filesender.belnet.be/download.php?token=7da0daea-1e5d-4004-953f-8884d2b976f7&files_ids=1461558' > belnet_file.zip
unzip belnet_file.zip
tar -xvf all_data.tar
tar -xvf example_results.tar -C ../results/
