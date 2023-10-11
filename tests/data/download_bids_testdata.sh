#!/bin/bash

[[ -f bidsdata.zip ]] || wget -O bidsdata.zip https://wustl.box.com/shared/static/2bzo8tgvagikbnjbdmcmejuy65g9kfra.zip
unzip -o bidsdata.zip
