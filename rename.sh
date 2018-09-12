#!/bin/bash

for filename in left-*.tiff; do
    if [[ "$filename" =~ ^left-0*([1-9]?[0-9]+)\.tiff$ ]]; then
        num=${BASH_REMATCH[1]}
        newname="left-$( printf '%06d' "$num" ).tiff"
        if [ "$filename" != "$newname" ] && [ ! -e "$newname" ]; then
            mv "$filename" "$newname"
        fi
    fi
done
