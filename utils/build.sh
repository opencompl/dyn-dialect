#!/bin/bash
cd build
ninja check-dyn
ninja tblgen-stats
ninja dialect-stats
