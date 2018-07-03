#/bin/sh

USAGE="usage: run.sh file_path userId_column_number itemId_column_number title_column_number"
test "$#" != "4" && echo "$USAGE" && exit 1

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

./script/extract.sh $1 $2 $3 $4
