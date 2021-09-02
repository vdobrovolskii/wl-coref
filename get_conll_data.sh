set -e  # any error will cause the script to exit immediately

downloads_dir=downloads
original_dir=$(pwd)

if [ $# -lt 2 ]; then
    echo -e "Usage:\n./get_conll_data.sh ONTONOTES_DIR DATA_DIR"
    exit
fi

if python -c "import sys; sys.exit(int(sys.version.startswith('2.7')))"; then
    echo "CONLL-2012 scripts require python 2.7!"
    echo "Activate python=2.7 environment and run the script again"
    exit
fi

if [ ! -d $1 ]; then
    echo "Invalid OntoNotes path!"
    exit
fi

ontonotes_dir=$1/data/files/data

if [ ! -d $ontonotes_dir ]; then
    echo "/data/files/data not found in OntoNotes path!"
    exit
fi

data_dir=$2

if [ ! -d $2 ]; then
    mkdir $2
fi

if [ ! -d $downloads_dir ]; then
    mkdir $downloads_dir
fi

cd $downloads_dir

curl -O https://cemantix.org/conll/2012/download/conll-2012-train.v4.tar.gz
curl -O https://cemantix.org/conll/2012/download/conll-2012-development.v4.tar.gz
curl -O https://cemantix.org/conll/2012/download/test/conll-2012-test-official.v9.tar.gz
curl -O https://cemantix.org/conll/2012/download/test/conll-2012-test-supplementary.v9.tar.gz
curl -O https://cemantix.org/conll/2012/download/test/conll-2012-test-key.tar.gz
curl -O https://cemantix.org/conll/2012/download/conll-2012-scripts.v3.tar.gz

cd $original_dir

for filename in $downloads_dir/conll-2012*.tar.gz; do
    tar -xvzf $filename -C $data_dir
done

$data_dir/conll-2012/v3/scripts/skeleton2conll.sh -D $ontonotes_dir $data_dir/conll-2012
