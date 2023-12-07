#!/bin/bash

[[ -f bidsdata.zip ]] || curl -L -o bidsdata.zip https://wustl.box.com/shared/static/2bzo8tgvagikbnjbdmcmejuy65g9kfra.zip
unzip -o bidsdata.zip
