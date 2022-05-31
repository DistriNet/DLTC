#!/bin/bash

curl 'https://filesender.belnet.be/download.php?token=56b46a01-7d0b-4dfe-b939-ab406b77bd45&files_ids=1190979%2C1190980' --output all_data.tar
tar -xvf all_data.tar
