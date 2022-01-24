#!/bin/bash

header_c="/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */"

header_s="# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited."

SRC_DIR=../
if [ -n "$1" ]
then
    SRC_DIR=$1
fi

for f in $(find "${SRC_DIR}" -name '*.cpp' -or -name '*.h'); do
    echo -n $f;
    if grep -q "Copyright (c)" $f; then
        echo " Copyright not needed"
    else
        echo " Copyright needed"
        echo -e "$header_c\n\n" > $f.new
        cat $f >> $f.new
        mv $f.new $f
    fi
done

#for f in $(find . -name 'Makefile*'); do
# for f in Makefile; do
for f in $(find "${SRC_DIR}" -name '*.py'); do
    echo -n $f;
    if grep -q "Copyright (c)" $f; then
        echo " Copyright not needed"
    else
        echo " Copyright needed"
        echo -e "$header_s\n\n" > $f.new
        cat $f >> $f.new
        mv $f.new $f
    fi
done

