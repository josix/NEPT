#! /bin/sh
# This script could extract two column (user and item)'s data from source file
# and export to file "user-item.data" which is separated by space.

USAGE="usage: extract.sh file_path userId_column_number itemId_column_number"
test "$#" != "3" && echo "$USAGE" && exit 1

if ! [ -e $1 ]
then
  echo "$USAGE\nFile \"$1\" is not exists"
  exit 1
fi

if ! [ -f $1 ]
then
  echo "$USAGE\n\"$1\" is not a file"
  exit 1
fi

awk -f "./extract.awk" -v user_column=$2 -v item_column=$3 $1 > ../data/user-item.data
