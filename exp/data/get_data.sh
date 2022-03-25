#!/bin/bash

curl 'https://filesender.belnet.be/download.php?token=57b4bc25-bde4-4fd1-be15-bf0a6e4ee797&files_ids=1056900%2C1056901' > belnet_file.zip
unzip belnet_file.zip
tar -xvf all_data.tar
tar -xvf example_results.tar -C ../results/
