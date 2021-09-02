set -e # any error will cause the script to exit immediately

git clone https://github.com/conll/reference-coreference-scorers.git

# downloading stanford parser

downloads_dir=downloads

if [ ! -d $downloads_dir ]; then
    mkdir $downloads_dir
fi

curl -o $downloads_dir/stanford-parser.jar https://raw.githubusercontent.com/DoodleJZ/HPSG-Neural-Parser/master/data/stanford-parser_3.3.0.jar
