#!/bin/bash

curl 'https://filesender.belnet.be/download.php?token=4f5b2172-f6e3-4d3d-bd8a-1ffda8ee4d16&files_ids=1039431%2C1039432%2C1039433%2C1039434%2C1039435%2C1039436' --output all_data.tar
tar -xvf all_data.tar
