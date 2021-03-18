#! /bin/bash

while getopts cn option
do
case "${option}"
in
c)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
# print the contents of the variable on screen
echo ${DIR}
#docker run -v ${DIR}:/root/ -w /root/ -it pff_adaptivity
docker run -v /Users/meenu/:/root/ -w /root/ -it pff_adaptivity
;;

n) 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
# print the contents of the variable on screen
echo ${DIR}
docker run -p 8888:8888 -v ${DIR}:/root/ -w /root/ pff_adaptivity_notebook
;;
esac
done





