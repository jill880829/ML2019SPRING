# type "bash retrieval.sh {output_path}" to execute

wget https://github.com/MortalHappiness/ml_final_url2content/releases/download/v1.0/url2content.json
mv url2content.json auxiliary_data/url2content.json

python ./src/jieba_whoosh.py $1